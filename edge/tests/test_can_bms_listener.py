"""Receive-only CAN BMS listener and the shared profile (plan Task 10).

The hard rule of this workstream is that the Pi never transmits on a BMS
bus. Two active masters is how a battery disconnects itself, so these tests
check the rule at every layer that could break it: no transmit path exists
in the class, the bus is opened receive-only, and the interface must ALREADY
be listen-only before the bus is opened at all.

That last check is a check and never a fix. A monitoring tool that silently
reconfigures a network interface shared with an inverter and a battery is
worse than one that refuses to start.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collector.adapters import victron_can_profile as profile
from collector.adapters.base import AdapterStatus
from collector.adapters.can_bms_listener import CANBMSListenerAdapter
from collector.adapters.registry import ADAPTER_REGISTRY

BATTERY = "70000000-0000-0000-0000-000000000001"

LISTEN_ONLY_OUTPUT = (
    "3: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP mode "
    "DEFAULT group default qlen 10\\n    link/can  promiscuity 0 minmtu 0 maxmtu 0 \\n"
    "    can state ERROR-ACTIVE (berr-counter tx 0 rx 0) restart-ms 0 \\n"
    "\\t  bitrate 500000 sample-point 0.875 \\n"
    "\\t  tq 50 prop-seg 17 phase-seg1 17 phase-seg2 5 sjw 1\\n"
    "\\t  listen-only on \\n"
)
NORMAL_OUTPUT = LISTEN_ONLY_OUTPUT.replace("listen-only on", "loopback off")


def _config(**extra: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "asset_id": BATTERY,
        "interface": "can0",
        "bitrate": 500000,
        "profile": "victron_generic",
    }
    config.update(extra)
    return config


def _message(arb_id: int, data: bytes) -> MagicMock:
    message = MagicMock()
    message.arbitration_id = arb_id
    message.data = data
    return message


def _mock_can() -> MagicMock:
    mock_can = MagicMock()
    mock_can.Bus.return_value = MagicMock()
    mock_can.AsyncBufferedReader.return_value = MagicMock()
    mock_can.Notifier.return_value = MagicMock()
    return mock_can


async def _start(adapter: CANBMSListenerAdapter, *, state: str, **extra: Any) -> bool:
    mock_can = _mock_can()
    with (
        patch.dict("sys.modules", {"can": mock_can}),
        patch.object(
            CANBMSListenerAdapter, "_interface_state", AsyncMock(return_value=state)
        ),
    ):
        return await adapter.initialize(_config(**extra))


# ── Listen-only is enforced before the bus is opened ────────────


@pytest.mark.asyncio
async def test_a_listen_only_interface_is_accepted() -> None:
    adapter = CANBMSListenerAdapter()
    assert await _start(adapter, state=LISTEN_ONLY_OUTPUT) is True
    assert adapter._status is AdapterStatus.CONNECTED


@pytest.mark.asyncio
async def test_an_interface_that_is_not_listen_only_is_refused() -> None:
    adapter = CANBMSListenerAdapter()
    mock_can = _mock_can()
    with (
        patch.dict("sys.modules", {"can": mock_can}),
        patch.object(
            CANBMSListenerAdapter,
            "_interface_state",
            AsyncMock(return_value=NORMAL_OUTPUT),
        ),
    ):
        assert await adapter.initialize(_config()) is False

    assert adapter._status is AdapterStatus.ERROR
    # The bus is never opened at all — the refusal comes first.
    mock_can.Bus.assert_not_called()


@pytest.mark.asyncio
async def test_an_unconfirmable_interface_is_refused_not_assumed_safe() -> None:
    """No iproute2, no permission, no interface — none of those is evidence."""
    adapter = CANBMSListenerAdapter()
    mock_can = _mock_can()
    with (
        patch.dict("sys.modules", {"can": mock_can}),
        patch.object(
            CANBMSListenerAdapter, "_interface_state", AsyncMock(return_value=None)
        ),
    ):
        assert await adapter.initialize(_config()) is False
    mock_can.Bus.assert_not_called()


@pytest.mark.asyncio
async def test_the_adapter_never_reconfigures_the_interface() -> None:
    """It shares that bus with an inverter and a battery."""
    adapter = CANBMSListenerAdapter()
    with patch("asyncio.create_subprocess_exec") as spawn:
        spawn.return_value = AsyncMock(
            communicate=AsyncMock(return_value=(LISTEN_ONLY_OUTPUT.encode(), b"")),
            returncode=0,
        )
        await CANBMSListenerAdapter._interface_state("can0")

    argv = list(spawn.call_args.args)
    assert argv[0] == "ip"
    assert "set" not in argv  # only 'ip -details link show <if>'
    assert argv[1:] == ["-details", "link", "show", "can0"]
    assert adapter is not None


def test_the_listen_only_check_reads_the_iproute_flag() -> None:
    assert CANBMSListenerAdapter._is_listen_only(LISTEN_ONLY_OUTPUT) is True
    assert CANBMSListenerAdapter._is_listen_only(NORMAL_OUTPUT) is False
    assert CANBMSListenerAdapter._is_listen_only("") is False
    assert CANBMSListenerAdapter._is_listen_only(None) is False


@pytest.mark.asyncio
async def test_the_check_can_be_disabled_only_by_writing_it_down() -> None:
    """An isolated bench rig is the one legitimate case, and it is explicit."""
    adapter = CANBMSListenerAdapter()
    mock_can = _mock_can()
    with (
        patch.dict("sys.modules", {"can": mock_can}),
        patch.object(
            CANBMSListenerAdapter, "_interface_state", AsyncMock(return_value=None)
        ),
    ):
        assert await adapter.initialize(_config(require_listen_only=False)) is True
    mock_can.Bus.assert_called_once()


@pytest.mark.asyncio
async def test_the_bus_is_opened_receive_only() -> None:
    adapter = CANBMSListenerAdapter()
    mock_can = _mock_can()
    with (
        patch.dict("sys.modules", {"can": mock_can}),
        patch.object(
            CANBMSListenerAdapter,
            "_interface_state",
            AsyncMock(return_value=LISTEN_ONLY_OUTPUT),
        ),
    ):
        await adapter.initialize(_config())

    kwargs = mock_can.Bus.call_args.kwargs
    assert kwargs["receive_own_messages"] is False
    assert kwargs["interface"] == "socketcan"
    assert kwargs["channel"] == "can0"
    assert kwargs["bitrate"] == 500000


@pytest.mark.asyncio
async def test_the_two_hundred_and_fifty_kilobit_variant_is_configurable() -> None:
    """Ingeteam profiles run at 250000."""
    adapter = CANBMSListenerAdapter()
    mock_can = _mock_can()
    with (
        patch.dict("sys.modules", {"can": mock_can}),
        patch.object(
            CANBMSListenerAdapter,
            "_interface_state",
            AsyncMock(return_value=LISTEN_ONLY_OUTPUT),
        ),
    ):
        await adapter.initialize(_config(bitrate=250000))
    assert mock_can.Bus.call_args.kwargs["bitrate"] == 250000


# ── There is no transmit path ───────────────────────────────────


def test_the_class_exposes_no_send_method() -> None:
    for forbidden in ("send", "send_periodic", "transmit", "write_frame"):
        assert not hasattr(CANBMSListenerAdapter, forbidden), forbidden


@pytest.mark.asyncio
async def test_write_always_refuses() -> None:
    adapter = CANBMSListenerAdapter()
    assert await adapter.write("battery.soc_pct", 50.0) is False
    assert adapter.capabilities().can_write is False
    assert adapter.capabilities().supported_write_metrics == []


@pytest.mark.asyncio
async def test_the_notifier_is_given_listeners_only() -> None:
    adapter = CANBMSListenerAdapter()
    mock_can = _mock_can()
    with (
        patch.dict("sys.modules", {"can": mock_can}),
        patch.object(
            CANBMSListenerAdapter,
            "_interface_state",
            AsyncMock(return_value=LISTEN_ONLY_OUTPUT),
        ),
    ):
        await adapter.initialize(_config())

    listeners = mock_can.Notifier.call_args.args[1]
    assert listeners == [mock_can.AsyncBufferedReader.return_value]


@pytest.mark.asyncio
async def test_a_poll_never_touches_the_bus_send_path() -> None:
    adapter = CANBMSListenerAdapter()
    mock_can = _mock_can()
    bus = mock_can.Bus.return_value
    reader = mock_can.AsyncBufferedReader.return_value
    reader.get_message = AsyncMock(
        side_effect=[_message(0x355, bytes([88, 0, 97, 0])), TimeoutError()]
    )
    with (
        patch.dict("sys.modules", {"can": mock_can}),
        patch.object(
            CANBMSListenerAdapter,
            "_interface_state",
            AsyncMock(return_value=LISTEN_ONLY_OUTPUT),
        ),
    ):
        await adapter.initialize(_config())
        await adapter.poll()

    bus.send.assert_not_called()
    bus.send_periodic.assert_not_called()


# ── Frame decoding, shared with BYD ─────────────────────────────


def _decode(arb_id: int, data: bytes) -> dict[str, float]:
    readings = profile.decode_fields(arb_id, data, BATTERY, datetime.now(UTC))
    return {reading.metric_name: reading.value for reading in readings}


def test_charge_limits_frame() -> None:
    # CVL 57.6 V, CCL 100.0 A, DCL 120.0 A, little-endian u16 at 0.1
    values = _decode(0x351, bytes([0x40, 0x02, 0xE8, 0x03, 0xB0, 0x04]))
    assert values["battery.charge_voltage_v"] == pytest.approx(57.6)
    assert values["battery.charge_current_limit_a"] == pytest.approx(100.0)
    assert values["battery.discharge_current_limit_a"] == pytest.approx(120.0)


def test_soc_and_soh_frame() -> None:
    values = _decode(0x355, bytes([88, 0, 97, 0]))
    assert values["battery.soc_pct"] == 88.0
    assert values["battery.soh_pct"] == 97.0


def test_voltage_current_temperature_frame_is_signed_where_it_should_be() -> None:
    # 51.20 V, -12.3 A, -5.0 degC
    values = _decode(0x356, bytes([0x00, 0x14, 0x85, 0xFF, 0xCE, 0xFF]))
    assert values["battery.voltage_v"] == pytest.approx(51.20)
    assert values["battery.current_a"] == pytest.approx(-12.3)
    assert values["battery.temperature_c"] == pytest.approx(-5.0)


def test_an_unknown_frame_decodes_to_nothing() -> None:
    assert _decode(0x123, bytes([1, 2, 3, 4, 5, 6, 7, 8])) == {}


def test_a_truncated_frame_emits_nothing_for_the_missing_field() -> None:
    """A short frame is a bus problem, not a reading of zero."""
    values = _decode(0x356, bytes([0x00, 0x14]))  # voltage only
    assert values["battery.voltage_v"] == pytest.approx(51.20)
    assert "battery.current_a" not in values
    assert "battery.temperature_c" not in values


def test_an_empty_frame_emits_nothing_at_all() -> None:
    assert _decode(0x351, b"") == {}


# ── The alarm frame, and the spec's wrong id ────────────────────


def test_the_alarm_frame_is_0x359_not_the_specs_0x35a() -> None:
    """No source binds an alarm frame to 0x35A; both use 0x359."""
    assert profile.ALARM_FRAME_ID == 0x359
    assert 0x35A not in profile.VICTRON_FRAMES


def test_no_alarm_bits_set_reports_a_clear_state() -> None:
    readings = profile.decode_alarm(
        bytes([0, 0, 0, 0, 1, 0x50, 0x4E]), BATTERY, datetime.now(UTC)
    )
    assert len(readings) == 1
    assert readings[0].metric_name == "battery.state"
    assert readings[0].value == 0.0


def test_any_set_bit_reports_a_flagged_state() -> None:
    """Coarse on purpose: the two sources disagree on what each bit means."""
    for data in (
        bytes([0x02, 0, 0, 0, 1, 0x50, 0x4E]),  # a byte-0 bit
        bytes([0, 0x80, 0, 0, 1, 0x50, 0x4E]),  # a byte-1 bit
        bytes([0, 0, 0x04, 0, 1, 0x50, 0x4E]),  # a warning bit
        bytes([0, 0, 0, 0x01, 1, 0x50, 0x4E]),
    ):
        readings = profile.decode_alarm(data, BATTERY, datetime.now(UTC))
        assert readings[0].value == 1.0, data.hex()


def test_the_alarm_reading_carries_its_raw_bytes_as_evidence() -> None:
    readings = profile.decode_alarm(
        bytes([0x02, 0x00, 0x04, 0x00, 1, 0x50, 0x4E]), BATTERY, datetime.now(UTC)
    )
    assert "02,00,04,00" in readings[0].raw_register


def test_bytes_beyond_the_first_four_are_not_read_as_alarms() -> None:
    """Byte 4 is a module number and bytes 5-6 are the 'PN' token."""
    readings = profile.decode_alarm(
        bytes([0, 0, 0, 0, 0xFF, 0x50, 0x4E]), BATTERY, datetime.now(UTC)
    )
    assert readings[0].value == 0.0


def test_a_short_alarm_frame_emits_nothing() -> None:
    assert profile.decode_alarm(bytes([0, 0]), BATTERY, datetime.now(UTC)) == []


# ── Identity frames are never telemetry ─────────────────────────


def test_identity_frames_decode_to_text() -> None:
    assert profile.decode_identity(0x35E, b"PYLON\x00\x00\x00") == (
        "manufacturer",
        "PYLON",
    )
    assert profile.decode_identity(0x35F, b"US2000\x00\x00") == ("model", "US2000")


def test_a_non_identity_frame_decodes_to_none() -> None:
    assert profile.decode_identity(0x355, bytes([88, 0, 97, 0])) is None


@pytest.mark.asyncio
async def test_identity_reaches_the_identity_block_and_not_the_readings() -> None:
    adapter = CANBMSListenerAdapter()
    mock_can = _mock_can()
    reader = mock_can.AsyncBufferedReader.return_value
    reader.get_message = AsyncMock(
        side_effect=[
            _message(0x35E, b"PYLON\x00\x00\x00"),
            _message(0x355, bytes([88, 0, 97, 0])),
            TimeoutError(),
        ]
    )
    with (
        patch.dict("sys.modules", {"can": mock_can}),
        patch.object(
            CANBMSListenerAdapter,
            "_interface_state",
            AsyncMock(return_value=LISTEN_ONLY_OUTPUT),
        ),
    ):
        await adapter.initialize(_config())
        readings = await adapter.poll()

    assert adapter.identity() == {"manufacturer": "PYLON"}
    assert {r.metric_name for r in readings} == {"battery.soc_pct", "battery.soh_pct"}


# ── Bounded draining and shutdown ───────────────────────────────


@pytest.mark.asyncio
async def test_a_poll_is_bounded_in_frames() -> None:
    """A CAN bus makes thousands of frames a second; the Pi has 256 MB."""
    adapter = CANBMSListenerAdapter()
    mock_can = _mock_can()
    reader = mock_can.AsyncBufferedReader.return_value
    reader.get_message = AsyncMock(return_value=_message(0x355, bytes([1, 0, 1, 0])))
    with (
        patch.dict("sys.modules", {"can": mock_can}),
        patch.object(
            CANBMSListenerAdapter,
            "_interface_state",
            AsyncMock(return_value=LISTEN_ONLY_OUTPUT),
        ),
    ):
        await adapter.initialize(_config())
        readings = await adapter.poll()

    # Two readings per frame, capped at the frame limit.
    assert len(readings) <= 400 * 2


@pytest.mark.asyncio
async def test_a_malformed_frame_does_not_end_the_poll() -> None:
    adapter = CANBMSListenerAdapter()
    mock_can = _mock_can()
    reader = mock_can.AsyncBufferedReader.return_value
    bad = MagicMock()
    bad.arbitration_id = 0x351
    bad.data = None  # will raise inside the parser
    reader.get_message = AsyncMock(
        side_effect=[bad, _message(0x355, bytes([88, 0, 97, 0])), TimeoutError()]
    )
    with (
        patch.dict("sys.modules", {"can": mock_can}),
        patch.object(
            CANBMSListenerAdapter,
            "_interface_state",
            AsyncMock(return_value=LISTEN_ONLY_OUTPUT),
        ),
    ):
        await adapter.initialize(_config())
        readings = await adapter.poll()

    assert {r.metric_name for r in readings} == {"battery.soc_pct", "battery.soh_pct"}


@pytest.mark.asyncio
async def test_poll_without_a_reader_returns_empty() -> None:
    assert await CANBMSListenerAdapter().poll() == []


@pytest.mark.asyncio
async def test_shutdown_stops_the_notifier_and_closes_the_bus() -> None:
    adapter = CANBMSListenerAdapter()
    mock_can = _mock_can()
    with (
        patch.dict("sys.modules", {"can": mock_can}),
        patch.object(
            CANBMSListenerAdapter,
            "_interface_state",
            AsyncMock(return_value=LISTEN_ONLY_OUTPUT),
        ),
    ):
        await adapter.initialize(_config())
        await adapter.shutdown()

    mock_can.Notifier.return_value.stop.assert_called_once()
    mock_can.Bus.return_value.shutdown.assert_called_once()
    assert adapter._status is AdapterStatus.DISCONNECTED


# ── Configuration and registration ──────────────────────────────


@pytest.mark.asyncio
async def test_an_unknown_profile_is_refused() -> None:
    adapter = CANBMSListenerAdapter()
    mock_can = _mock_can()
    with patch.dict("sys.modules", {"can": mock_can}):
        assert await adapter.initialize(_config(profile="pylontech_v3")) is False
    mock_can.Bus.assert_not_called()


@pytest.mark.asyncio
async def test_a_missing_asset_id_is_refused() -> None:
    """An unattributed battery reading has nowhere to land."""
    adapter = CANBMSListenerAdapter()
    mock_can = _mock_can()
    with patch.dict("sys.modules", {"can": mock_can}):
        assert await adapter.initialize(_config(asset_id="")) is False


def test_both_profiles_are_selectable() -> None:
    assert set(profile.PROFILES) == {"victron_generic", "byd"}


def test_the_listener_is_registered() -> None:
    assert ADAPTER_REGISTRY["can_bms_listener"] is CANBMSListenerAdapter


def test_the_byd_adapter_now_shares_the_profile_table() -> None:
    """Extracted, not re-transcribed: the BYD contract must not move."""
    from collector.adapters.byd_can import CAN_MESSAGES

    assert CAN_MESSAGES is profile.VICTRON_FRAMES
    assert CAN_MESSAGES[0x356]["fields"]["battery.current_a"]["signed"] is True
    assert CAN_MESSAGES[0x356]["fields"]["battery.voltage_v"]["scale"] == 0.01


def test_the_profile_advertises_what_it_can_produce() -> None:
    metrics = profile.supported_metrics()
    assert "battery.soc_pct" in metrics
    assert "battery.state" in metrics
    assert "battery.charge_current_limit_a" in metrics
    # No cell metrics: those are BYD's own extension, not the shared profile.
    assert not any("cell" in metric for metric in metrics)


def test_no_freedomwon_specific_adapter_was_invented() -> None:
    """WS3 is built to the profile only until a prospect has the hardware."""
    assert "freedomwon" not in ADAPTER_REGISTRY
    assert not any("freedom" in name.lower() for name in ADAPTER_REGISTRY)


# ── The two refusal causes are distinguishable ──────────────────
#
# Both refuse, identically. But one sends a technician to the image build and
# the other to the interface, and a log line that names the wrong one costs a
# site visit: they re-run the ip command they already ran, restart, and read
# the same message.


@pytest.mark.asyncio
async def test_a_missing_ip_binary_reports_a_build_problem_not_a_wiring_one() -> None:
    from collector.adapters.can_bms_listener import IP_NOT_FOUND

    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        state = await CANBMSListenerAdapter._interface_state("can0")

    assert state == IP_NOT_FOUND
    assert CANBMSListenerAdapter._is_listen_only(state) is False


@pytest.mark.asyncio
async def test_a_missing_ip_binary_still_refuses_to_open_the_bus() -> None:
    """The behaviour is identical; only the diagnosis differs."""
    from collector.adapters.can_bms_listener import IP_NOT_FOUND

    adapter = CANBMSListenerAdapter()
    mock_can = _mock_can()
    with (
        patch.dict("sys.modules", {"can": mock_can}),
        patch.object(
            CANBMSListenerAdapter,
            "_interface_state",
            AsyncMock(return_value=IP_NOT_FOUND),
        ),
    ):
        assert await adapter.initialize(_config()) is False
    mock_can.Bus.assert_not_called()


@pytest.mark.asyncio
async def test_a_permission_error_is_still_an_unconfirmed_refusal() -> None:
    """Not the same as a missing binary: 'ip' exists but would not answer."""
    with patch("asyncio.create_subprocess_exec", side_effect=PermissionError):
        assert await CANBMSListenerAdapter._interface_state("can0") is None


@pytest.mark.asyncio
async def test_a_wedged_ip_command_is_killed_rather_than_left_behind() -> None:
    process = MagicMock()
    process.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
    process.kill = MagicMock()
    process.returncode = None
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
        assert await CANBMSListenerAdapter._interface_state("can0") is None
    process.kill.assert_called_once()


@pytest.mark.asyncio
async def test_a_nonzero_exit_is_an_unconfirmed_refusal() -> None:
    """An interface that does not exist yet exits non-zero."""
    process = MagicMock()
    process.communicate = AsyncMock(return_value=(b"", b"Device does not exist"))
    process.returncode = 1
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
        assert await CANBMSListenerAdapter._interface_state("can0") is None


def test_the_collector_image_can_actually_run_the_check() -> None:
    """The listen-only gate shells out to `ip`; the image has to contain it.

    Without this the adapter is correct and permanently dead: it refuses on
    every node because it can never confirm what it is required to confirm.
    """
    from pathlib import Path

    dockerfile = Path("collector/Dockerfile.arm64").read_text()
    assert "iproute2" in dockerfile
