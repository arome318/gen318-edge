"""The Voltronic adapter, against a fake inverter that answers real frames.

The checksum is the thing everything else rests on, so it is pinned to two
pieces of external ground truth BEFORE it is used to build any fixture:

  1. The published check value for CRC-16/XMODEM over ASCII "123456789"
     (0x31C3), which is not from any of this protocol's sources.
  2. The two literal PI18 acknowledgement frames that appear as byte strings
     in mpp-solar's own source — b"^1\\x0b\\xc2\\r" and b"^0\\x1b\\xe3\\r".
     Those bytes were written by someone reading real hardware, and they
     reproduce only if the substitution rule is right as well as the
     polynomial.

After that the fake inverter frames its replies with the same function the
adapter verifies with, which is safe precisely because that function has
already been checked against something outside the codebase.
"""

from __future__ import annotations

import asyncio
import time
import math
from typing import Any

import pytest

from collector.adapters.registry import ADAPTER_REGISTRY
from collector.adapters.voltronic_inverter import (
    HidRawTransport,
    SerialTransport,
    VoltronicInverterAdapter,
    build_transport,
)
from collector.adapters.voltronic_protocol import (
    PI18_ACK,
    PI18_COMMANDS,
    PI18_NAK,
    PI30_COMMANDS,
    READ_COMMANDS,
    VoltronicProtocolError,
    crc16_xmodem,
    crc_bytes,
    frame_pi18,
    frame_pi30,
    parse_pi30_reply,
    parse_qmod,
    parse_qpigs,
    parse_qpiws,
)

ASSETS = {
    "inverter": "11111111-1111-1111-1111-111111111111",
    "battery": "22222222-2222-2222-2222-222222222222",
    "solar_array": "33333333-3333-3333-3333-333333333333",
    "grid_connection": "44444444-4444-4444-4444-444444444444",
}


# ── The checksum, pinned to ground truth outside this repo ──────


def test_the_checksum_matches_the_published_xmodem_check_value() -> None:
    assert crc16_xmodem(b"123456789") == 0x31C3


def test_the_substitution_rule_reproduces_the_two_captured_ack_frames() -> None:
    """These byte literals came off real hardware, not off a spec."""
    high, low = crc_bytes(b"^1")
    assert bytes((ord("^"), ord("1"), high, low, 0x0D)) == PI18_ACK
    high, low = crc_bytes(b"^0")
    assert bytes((ord("^"), ord("0"), high, low, 0x0D)) == PI18_NAK


def test_no_allowed_command_lands_on_the_disputed_substitution() -> None:
    """The one open disagreement in the protocol cannot reach this adapter.

    mpp-solar bumps a checksum byte of 0x00; the C reference does not. The
    module implements only the three bytes both agree on. That is sound as
    long as no command we send produces a 0x00 checksum byte — so assert it,
    and this test fails the day someone adds a command that does, instead of
    the inverter quietly rejecting it on site.
    """
    offenders = []
    for command in sorted(PI30_COMMANDS):
        high, low = crc_bytes(command.encode())
        if high == 0 or low == 0:
            offenders.append(command)
    for command, prefix in sorted(PI18_COMMANDS.items()):
        high, low = crc_bytes((prefix + command).encode())
        if high == 0 or low == 0:
            offenders.append(prefix + command)
    assert not offenders, (
        f"{offenders} produce a 0x00 checksum byte, where mpp-solar and the C "
        f"reference disagree on whether to substitute. Resolve it before shipping."
    )


# ── Read-only by construction ───────────────────────────────────


@pytest.mark.parametrize(
    "setter",
    ["PBT", "POP", "DAT", "PE", "PD", "F", "PCP", "MCHGC", "PSDV", "PBATCD"],
)
def test_pi30_setters_are_refused(setter: str) -> None:
    """These change battery chemistry, charger priority and the clock."""
    with pytest.raises(VoltronicProtocolError):
        frame_pi30(setter)


def test_the_pi18_setter_envelope_is_never_framed() -> None:
    assert all(prefix.startswith("^P") for prefix in PI18_COMMANDS.values())
    for command in ("POP", "PBT", "MCHGV", "BUCD"):
        with pytest.raises(VoltronicProtocolError):
            frame_pi18(command)


def test_the_allowlist_holds_only_queries() -> None:
    # Every PI30 entry is a Q-command; PI18 names its queries differently but
    # all of them are carried by the ^P envelope, asserted above.
    assert all(c.startswith("Q") for c in PI30_COMMANDS)
    assert READ_COMMANDS == PI30_COMMANDS | frozenset(PI18_COMMANDS)


@pytest.mark.asyncio
async def test_write_is_refused_and_sends_nothing() -> None:
    inverter = FakeInverter(generation="pi30")
    adapter = await _connected(inverter)
    before = len(inverter.seen)
    assert await adapter.write("battery.power_setpoint_w", 500.0) is False
    assert len(inverter.seen) == before
    assert adapter.capabilities().can_write is False
    assert adapter.capabilities().supported_write_metrics == []


# ── A fake inverter that answers properly framed replies ────────


def _pi30(payload: str) -> bytes:
    body = b"(" + payload.encode("ascii")
    high, low = crc_bytes(body)
    return body + bytes((high, low, 0x0D))


def _pi18(payload: str) -> bytes:
    body = f"^D{len(payload) + 3:03d}".encode() + payload.encode("ascii")
    high, low = crc_bytes(body)
    return body + bytes((high, low, 0x0D))


#: The QPIGS fixture from mpp-solar's own pi30.py, 21 fields.
QPIGS_LONG = (
    "000.0 00.0 230.0 49.9 0161 0119 003 460 57.50 012 100 0069 0014 "
    "103.8 57.45 00000 00110110 00 00 00856 010"
)
#: The same reading truncated to the 17 fields the 2014 vendor document
#: defines. A unit on older firmware sends exactly this.
QPIGS_SHORT = " ".join(QPIGS_LONG.split(" ")[:17])

#: PI18 GS: 28 comma-separated integers, tenths where the field is scaled.
PI18_GS = (
    "2323,499,2298,500,0805,0789,032,532,533,000,000,015,088,041,"
    "035,036,0412,0388,2451,2398,0,2,2,1,1,2,1,0"
)


class FakeInverter:
    """Answers the commands this adapter is allowed to send, and no others."""

    def __init__(
        self,
        *,
        generation: str,
        qpigs: str = QPIGS_LONG,
        corrupt: bool = False,
        nak: bool = False,
        silent: bool = False,
    ) -> None:
        self.generation = generation
        self.qpigs = qpigs
        self.corrupt = corrupt
        self.nak = nak
        self.silent = silent
        self.seen: list[bytes] = []

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def exchange(self, frame: bytes, timeout: float) -> bytes:
        self.seen.append(frame)
        if self.silent:
            return b""
        text = frame[:-3].decode("ascii", errors="replace")
        reply = self._reply(text)
        if reply is None:
            return b""
        if self.nak:
            return _pi30("NAK") if self.generation == "pi30" else PI18_NAK
        if self.corrupt:
            return reply[:-3] + b"\x00\x00" + reply[-1:]
        return reply

    def _reply(self, text: str) -> bytes | None:
        if self.generation == "pi30":
            if text.startswith("^"):
                return None  # a PI30 unit ignores the PI18 envelope
            return {
                "QPI": _pi30("PI30"),
                "QID": _pi30("9293333010501"),
                "QVFW": _pi30("VERFW:00072.70"),
                "QPIGS": _pi30(self.qpigs),
                "QMOD": _pi30("B"),
                "QPIWS": _pi30("00000100000000001000000000000000"),
            }.get(text)
        if not text.startswith("^P"):
            return None  # a PI18 unit ignores the bare PI30 command
        return {
            "^P005PI": _pi18("18"),
            "^P005ID": _pi18("92932105105335005535"),
            "^P006VFW": _pi18("00072.70"),
            "^P005GS": _pi18(PI18_GS),
            "^P006MOD": _pi18("3"),
        }.get(text)


async def _connected(
    inverter: FakeInverter, **overrides: Any
) -> VoltronicInverterAdapter:
    adapter = VoltronicInverterAdapter()
    config: dict[str, Any] = {
        "port": "/dev/ttyUSB0",
        "asset_mappings": dict(ASSETS),
        "timeout_seconds": 1.0,
    }
    config.update(overrides)

    # initialize() builds its own transport; substitute the fake by patching
    # the builder for the duration of the call so the rest of initialize —
    # detection, identity, status — runs for real.
    import collector.adapters.voltronic_inverter as module

    saved = module.build_transport
    module.build_transport = lambda cfg: inverter  # type: ignore[assignment]
    try:
        assert await adapter.initialize(config) is True
    finally:
        module.build_transport = saved  # type: ignore[assignment]
    return adapter


# ── Detection ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_pi30_unit_is_detected_and_identified() -> None:
    inverter = FakeInverter(generation="pi30")
    adapter = await _connected(inverter)
    assert adapter.identity()["protocol"] == "pi30"
    assert adapter.identity()["serial"] == "9293333010501"
    assert adapter.identity()["firmware"] == "VERFW:00072.70"


@pytest.mark.asyncio
async def test_a_pi18_unit_is_detected_after_pi30_goes_unanswered() -> None:
    inverter = FakeInverter(generation="pi18")
    adapter = await _connected(inverter)
    assert adapter.identity()["protocol"] == "pi18"
    # The bare PI30 probe was tried first and ignored, then the PI18 one.
    assert inverter.seen[0].startswith(b"QPI")
    assert inverter.seen[1].startswith(b"^P005PI")


async def _start(
    inverter: FakeInverter, **overrides: Any
) -> tuple[VoltronicInverterAdapter, bool]:
    """initialize() against the fake transport, returning what it returned."""
    adapter = VoltronicInverterAdapter()
    config: dict[str, Any] = {
        "port": "/dev/ttyUSB0",
        "asset_mappings": dict(ASSETS),
        "timeout_seconds": 0.1,
    }
    config.update(overrides)
    import collector.adapters.voltronic_inverter as module

    saved = module.build_transport
    module.build_transport = lambda cfg: inverter  # type: ignore[assignment]
    try:
        ok = await adapter.initialize(config)
    finally:
        module.build_transport = saved  # type: ignore[assignment]
    return adapter, ok


@pytest.mark.asyncio
async def test_a_silent_inverter_still_starts_and_recovers_when_it_wakes() -> None:
    """The adapter must not strand itself on a unit that was merely asleep.

    This replaces a test that asserted initialize() returns False here. That
    was wrong, and wrong in a way only visible from the collector: an adapter
    whose initialize returns False never gets a poll task registered, AND the
    reconnect command cannot reach it either, because that command only looks
    at adapters holding a live instance. So a Pi that power-cycles at 03:00
    while the inverter is asleep would never read that inverter again until
    somebody restarted the container — on a site whose whole point is running
    unattended.

    A Modbus adapter does not behave that way: its connect() only needs the
    port to open, so a silent-but-present device still gets polled and heals
    itself when it wakes. This now matches.
    """
    inverter = FakeInverter(generation="pi30", silent=True)
    adapter, ok = await _start(inverter)

    assert ok is True, "a silent unit must not stop the adapter starting"
    assert await adapter.poll() == [], "it must publish nothing while unidentified"
    assert adapter.identity().get("protocol", "") == ""

    # The inverter wakes. The next poll picks it up, with no restart.
    inverter.silent = False
    values = {r.metric_name: r.value for r in await adapter.poll()}
    assert values["battery.voltage_v"] == pytest.approx(57.50)
    assert adapter.identity()["protocol"] == "pi30"
    assert adapter.identity()["serial"] == "9293333010501"
    await adapter.shutdown()


@pytest.mark.asyncio
async def test_a_port_that_will_not_open_does_fail_to_initialize() -> None:
    """The distinction that matters: no port at all is still a failure."""

    class _DeadPort(FakeInverter):
        async def open(self) -> None:
            raise OSError("no such device")

    _adapter, ok = await _start(_DeadPort(generation="pi30"))
    assert ok is False


@pytest.mark.asyncio
async def test_missing_asset_mappings_refuse_to_initialize() -> None:
    adapter = VoltronicInverterAdapter()
    assert await adapter.initialize({"port": "/dev/ttyUSB0"}) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad", ["axpert-1", "", "inverter", "9293333010501", "not-a-uuid-at-all"]
)
async def test_a_binding_that_is_not_a_uuid_refuses_to_start(bad: str) -> None:
    """Starting on one of these loses the inverter's whole series, silently.

    The adapter would log CONNECTED and collect happily. The rows would reach
    SQLite, fail sync's non-UUID pre-flight, be marked rejected and pruned
    after the retention window — and the only trace would be a warning in a
    different container. Every other role-binding adapter here refuses to
    start on this; so does this one.
    """
    mappings = dict(ASSETS)
    mappings["battery"] = bad
    _adapter, ok = await _start(
        FakeInverter(generation="pi30"), asset_mappings=mappings
    )
    assert ok is False


# ── Polling ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_pi30_poll_decodes_the_reference_fixture() -> None:
    adapter = await _connected(FakeInverter(generation="pi30"))
    values = {r.metric_name: r.value for r in await adapter.poll()}

    assert values["grid.voltage_v"] == pytest.approx(0.0)  # running off battery
    assert values["grid.frequency_hz"] == pytest.approx(0.0)
    assert values["inverter.ac_output_voltage_v"] == pytest.approx(230.0)
    assert values["inverter.ac_output_freq_hz"] == pytest.approx(49.9)
    assert values["inverter.ac_output_power_w"] == pytest.approx(119.0)
    assert values["battery.voltage_v"] == pytest.approx(57.50)
    assert values["battery.soc_pct"] == pytest.approx(100.0)
    assert values["inverter.temperature_c"] == pytest.approx(69.0)
    assert values["solar.pv_voltage_v"] == pytest.approx(103.8)
    assert values["solar.pv_power_w"] == pytest.approx(856.0)
    # 012 A charging, 00000 A discharging: positive is charge.
    assert values["battery.current_a"] == pytest.approx(12.0)
    assert values["inverter.state"] == pytest.approx(3.0)  # QMOD "B" = battery


@pytest.mark.asyncio
async def test_the_disputed_qpigs_position_is_never_published() -> None:
    """Index 12 is amps in one source and watts in another at the same slot."""
    values = parse_qpigs(tuple(QPIGS_LONG.split(" ")))
    # 0014 at index 12 must not appear as any metric's value.
    assert 14.0 not in values.values()
    assert "solar.pv_current_a" not in values


@pytest.mark.asyncio
async def test_an_older_firmware_loses_pv_power_and_keeps_everything_else() -> None:
    """The 2014 document's 17-field reply has no PV power field at all."""
    adapter = await _connected(FakeInverter(generation="pi30", qpigs=QPIGS_SHORT))
    values = {r.metric_name: r.value for r in await adapter.poll()}
    assert "solar.pv_power_w" not in values
    assert values["battery.voltage_v"] == pytest.approx(57.50)
    assert values["battery.current_a"] == pytest.approx(12.0)


@pytest.mark.asyncio
async def test_a_pi18_poll_applies_the_tenths_scaling() -> None:
    adapter = await _connected(FakeInverter(generation="pi18"))
    values = {r.metric_name: r.value for r in await adapter.poll()}
    assert values["grid.voltage_v"] == pytest.approx(232.3)
    assert values["grid.frequency_hz"] == pytest.approx(49.9)
    assert values["inverter.ac_output_voltage_v"] == pytest.approx(229.8)
    assert values["battery.voltage_v"] == pytest.approx(53.2)
    assert values["battery.soc_pct"] == pytest.approx(88.0)
    assert values["solar.mppt1_power_w"] == pytest.approx(412.0)
    assert values["solar.mppt2_power_w"] == pytest.approx(388.0)
    assert values["solar.pv_power_w"] == pytest.approx(800.0)
    assert values["solar.mppt1_voltage_v"] == pytest.approx(245.1)
    # 015 A charging, 000 A discharging.
    assert values["battery.current_a"] == pytest.approx(15.0)
    assert values["inverter.state"] == pytest.approx(3.0)  # MOD 3 = battery


@pytest.mark.asyncio
async def test_readings_are_routed_to_the_right_asset() -> None:
    adapter = await _connected(FakeInverter(generation="pi30"))
    by_asset = {r.metric_name: r.asset_id for r in await adapter.poll()}
    assert by_asset["battery.soc_pct"] == ASSETS["battery"]
    assert by_asset["grid.voltage_v"] == ASSETS["grid_connection"]
    assert by_asset["solar.pv_power_w"] == ASSETS["solar_array"]
    assert by_asset["inverter.ac_output_voltage_v"] == ASSETS["inverter"]


@pytest.mark.asyncio
async def test_an_unbound_role_drops_its_readings_rather_than_misfiling_them() -> None:
    inverter = FakeInverter(generation="pi30")
    adapter = await _connected(
        inverter,
        asset_mappings={k: v for k, v in ASSETS.items() if k != "solar_array"},
    )
    metrics = {r.metric_name for r in await adapter.poll()}
    assert not any(m.startswith("solar.") for m in metrics)
    assert "battery.soc_pct" in metrics


# ── Refusing what it cannot trust ───────────────────────────────


@pytest.mark.asyncio
async def test_a_corrupt_checksum_yields_no_readings() -> None:
    adapter = await _connected(FakeInverter(generation="pi30"))
    adapter._transport.corrupt = True  # type: ignore[union-attr]
    assert await adapter.poll() == []


@pytest.mark.asyncio
async def test_a_nak_yields_no_readings() -> None:
    adapter = await _connected(FakeInverter(generation="pi30"))
    adapter._transport.nak = True  # type: ignore[union-attr]
    assert await adapter.poll() == []


@pytest.mark.asyncio
async def test_a_silent_inverter_yields_no_readings_and_never_raises() -> None:
    adapter = await _connected(FakeInverter(generation="pi30"))
    adapter._transport.silent = True  # type: ignore[union-attr]
    assert await adapter.poll() == []


def test_a_reply_missing_its_opening_paren_is_rejected() -> None:
    body = b"230.0 49.9"
    high, low = crc_bytes(body)
    with pytest.raises(VoltronicProtocolError):
        parse_pi30_reply(body + bytes((high, low, 0x0D)))


def test_an_unmapped_mode_letter_emits_nothing_rather_than_a_guess() -> None:
    assert parse_qmod(("Z",)) == {}
    assert parse_qmod(("B",)) == {"inverter.state": 3.0}


def test_the_warning_word_reports_the_first_set_bit_and_zero_for_none() -> None:
    assert parse_qpiws(("00000100000000001000000000000000",)) == {
        "inverter.error_code": 6.0
    }
    assert parse_qpiws(("0" * 32,)) == {"inverter.error_code": 0.0}
    # V1's own fixtures are 34 and 38 characters against a 32-bit table, so
    # anything past bit 32 is ignored rather than trusted.
    assert parse_qpiws(("0" * 36 + "1",)) == {"inverter.error_code": 0.0}


# ── Transport selection ─────────────────────────────────────────


def test_the_device_path_picks_the_transport() -> None:
    assert isinstance(build_transport({"port": "/dev/ttyUSB0"}), SerialTransport)
    assert isinstance(build_transport({"port": "/dev/hidraw0"}), HidRawTransport)
    assert isinstance(
        build_transport({"port": "/dev/ttyACM0", "transport": "hidraw"}),
        HidRawTransport,
    )


def test_a_missing_port_is_refused() -> None:
    with pytest.raises(ValueError):
        build_transport({})


def test_the_serial_transport_can_be_constructed_with_the_backend_installed() -> None:
    """The pyserial defect again: construction is where it would raise."""
    import serial

    transport = SerialTransport("/dev/ttyUSB0", 2400)
    assert transport._baudrate == 2400
    assert serial.EIGHTBITS == 8


def test_the_adapter_is_in_the_registry() -> None:
    assert ADAPTER_REGISTRY["voltronic_inverter"] is VoltronicInverterAdapter


@pytest.mark.asyncio
async def test_shutdown_is_idempotent() -> None:
    adapter = await _connected(FakeInverter(generation="pi30"))
    await adapter.shutdown()
    await adapter.shutdown()
    assert await adapter.poll() == []
    assert not asyncio.get_running_loop().is_closed()


# ── Transport lifetime: the descriptor must not outlive its worker ──


class _SlowPort:
    """A port that answers slowly, and counts what it is asked after close."""

    def __init__(self) -> None:
        self.closed = False
        self.reads = 0
        self.reads_after_close = 0

    def reset_input_buffer(self) -> None:
        return None

    def write(self, data: bytes) -> int:
        return len(data)

    def flush(self) -> None:
        return None

    def read(self, size: int) -> bytes:
        self.reads += 1
        if self.closed:
            self.reads_after_close += 1
        time.sleep(0.02)
        return b""

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_closing_waits_for_the_device_thread_before_freeing_the_handle() -> None:
    """A closed descriptor is reissued to the very next open.

    So a worker still inside its read loop must not be able to issue another
    syscall after close() returns: on this node the number it is holding
    could by then belong to the SQLite handle. Cancelling the poll is what
    makes this reachable — the await unblocks, but the thread does not stop,
    and the collector's reconnect path cancels and reopens back to back.
    """
    port = _SlowPort()
    transport = SerialTransport("/dev/ttyUSB0", 2400)
    transport._serial = port

    task = asyncio.create_task(transport.exchange(b"QPI\x00\x00\r", 5.0))
    # The worker settles for _SETTLE_SECONDS after writing, so wait past it.
    await asyncio.sleep(0.45)  # let the worker get inside its read loop
    assert port.reads > 0, "expected the worker to be reading"
    assert not transport._idle.is_set()

    task.cancel()  # the collector cancelling a poll, mid-read
    with pytest.raises(asyncio.CancelledError):
        await task

    # The await is finished; the THREAD is not. If the idle flag were set by
    # the coroutine rather than by the worker itself, this is where it would
    # lie, and close() would free a descriptor that is still in use.
    assert not transport._idle.is_set(), (
        "the device was reported idle while its worker thread was still running"
    )

    await transport.close()

    assert transport._idle.is_set(), "close() returned while a worker was live"
    assert port.closed, "the port should be closed once the worker has left"
    assert port.reads_after_close == 0, (
        f"{port.reads_after_close} read(s) reached the device after it was "
        "closed; a reissued descriptor would carry those to another file"
    )


@pytest.mark.asyncio
async def test_an_exchange_is_refused_once_the_transport_is_closing() -> None:
    transport = SerialTransport("/dev/ttyUSB0", 2400)
    transport._serial = _SlowPort()
    await transport.close()
    assert await transport.exchange(b"QPI\x00\x00\r", 0.1) == b""


@pytest.mark.asyncio
async def test_the_hidraw_worker_never_captures_the_descriptor() -> None:
    """Re-reading it each iteration is what lets close() stop the worker."""
    import inspect

    source = inspect.getsource(HidRawTransport.exchange)
    assert source.count("fd = self._fd") >= 3, (
        "the hidraw worker must re-read the descriptor before each syscall, "
        "not bind it once at the top"
    )


def test_a_non_finite_field_is_not_a_reading() -> None:
    """float() accepts 'nan' and 'inf'; sync tombstones both."""
    fields = tuple("000.0 00.0 nan 49.9 0161 0119 003 460 inf 012 100".split(" "))
    values = parse_qpigs(fields)
    assert "inverter.ac_output_voltage_v" not in values  # was "nan"
    assert "battery.voltage_v" not in values  # was "inf"
    assert values["grid.voltage_v"] == pytest.approx(0.0)
    assert all(math.isfinite(v) for v in values.values())


# ── Tier-3 evidence: the unit's identity has to leave the node ──


@pytest.mark.asyncio
async def test_the_unit_identity_ships_as_evidence_like_every_other_adapter() -> None:
    """Otherwise a Voltronic validation record is weaker than every other.

    The collector ships a local adapter's provenance and device identity as
    one tier-3 event, so a record can say which code, on what evidence, read
    which unit. That hook is named for vendor maps; this adapter has none,
    but a record needs the same facts regardless.
    """
    adapter = await _connected(FakeInverter(generation="pi30"))
    summary = adapter.vendor_map_summary()
    assert summary is not None

    assert summary["vendor"] == "voltronic"
    assert summary["verification"] == "community", "never more than the maps claim"
    assert "2026-09-19" in summary["source"]
    assert summary["register_map"] is None, "this adapter has no register map"
    assert summary["protocol"] == "pi30"
    assert summary["identity"]["serial"] == "9293333010501"
    assert summary["identity"]["firmware"] == "VERFW:00072.70"
    await adapter.shutdown()


@pytest.mark.asyncio
async def test_an_unidentified_unit_ships_no_evidence_yet() -> None:
    """A silent inverter has nothing to say about itself; don't invent it."""
    adapter, ok = await _start(FakeInverter(generation="pi30", silent=True))
    assert ok is True
    assert adapter.vendor_map_summary() is None
    await adapter.shutdown()


@pytest.mark.asyncio
async def test_the_collector_hook_accepts_this_adapters_summary() -> None:
    """The contract is duck-typed, so assert the shape the collector reads."""
    adapter = await _connected(FakeInverter(generation="pi18"))
    summary = adapter.vendor_map_summary()
    assert summary is not None
    # collector.main._queue_vendor_map_evidence reads these two by name.
    assert isinstance(summary.get("name"), str) and summary["name"]
    assert isinstance(summary.get("verification"), str)
    import json

    json.dumps(summary)  # it is enqueued as JSON; must be serialisable
    await adapter.shutdown()
