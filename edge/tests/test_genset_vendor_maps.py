"""DSE GenComm map and the ComAp template (plan Task 8).

DSE is the best-sourced family in the tranche: three public sources that
agree register by register, plus DSE's own documented sentinel convention.
That convention is what these tests protect hardest — a controller with no
oil-pressure sender writes 0xFFFF, and publishing 65535 kPa as a measurement
is exactly the failure the whole vendor-map design exists to prevent.

ComAp is the opposite case. Its register-to-object table is exported per
controller, so the shipped file is a TEMPLATE with blank addresses that
refuses to poll. A half-filled export must never sweep and report a clean
run over nothing.
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import pytest

from collector.adapters.generic_modbus import GenericModbusAdapter
from collector.adapters.vendor_map import (
    VendorMapError,
    load_vendor_map,
    resolve_vendor_map,
)

GENSET = "50000000-0000-0000-0000-000000000001"
ROLES = {"diesel_genset": GENSET}


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _u32_be(value: int) -> tuple[int, int]:
    """DSE puts the HIGH word at the LOWER address — the opposite of Deye."""
    return (value >> 16) & 0xFFFF, value & 0xFFFF


def _synthetic_dse() -> list[int]:
    """A plausible DSE7320 register image. SYNTHETIC — never a captured dump."""
    words = [0] * 2000

    words[772] = 2  # control mode
    words[1408] = 3  # engine operating state

    words[1024] = 420  # 420 kPa oil pressure
    words[1025] = 85  # 85 degC coolant
    words[1027] = 72  # 72 % fuel
    words[1029] = 261  # 26.1 V starting battery
    words[1031] = 500  # 50.0 Hz
    words[1032], words[1033] = _u32_be(2305)  # 230.5 V

    words[1536], words[1537] = _u32_be(184000)  # 184 kW as WATTS
    words[1558] = 736  # 73.6 % load

    words[1798], words[1799] = _u32_be(36000)  # 36000 s = 10.0 h
    words[1800], words[1801] = _u32_be(45678)  # 4567.8 kWh
    words[1808], words[1809] = _u32_be(1234)  # 1234 starts
    return words


class _Server:
    def __init__(self, words: list[int]) -> None:
        self.words = words
        self.port = _free_port()
        self._server: Any = None
        self._task: asyncio.Task[Any] | None = None

    async def __aenter__(self) -> _Server:
        from pymodbus.server import ModbusTcpServer
        from pymodbus.simulator import DataType, SimData, SimDevice

        block = SimData(0, values=list(self.words), datatype=DataType.REGISTERS)
        self._server = ModbusTcpServer(
            SimDevice(10, simdata=[block]), address=("127.0.0.1", self.port)
        )
        self._task = asyncio.create_task(self._server.serve_forever())
        await asyncio.sleep(0.3)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._server is not None:
            await self._server.shutdown()
        if self._task is not None:
            self._task.cancel()


async def _poll(words: list[int], **config: Any) -> dict[str, float]:
    async with _Server(words) as server:
        adapter = GenericModbusAdapter()
        base: dict[str, Any] = {
            "vendor_map": "dse_gencomm",
            "protocol": "tcp",
            "host": "127.0.0.1",
            "port": server.port,
            "unit_id": 10,
            "asset_mappings": dict(ROLES),
        }
        base.update(config)
        assert await adapter.initialize(base) is True
        readings = await adapter.poll()
        await adapter.shutdown()
    return {r.metric_name: r.value for r in readings}


# ── DSE decodes as its own specification states ─────────────────


@pytest.mark.asyncio
async def test_the_dse_map_polls_a_real_server() -> None:
    values = await _poll(_synthetic_dse())

    assert values["genset.oil_pressure_kpa"] == 420.0
    assert values["genset.coolant_temperature_c"] == 85.0
    assert values["genset.fuel_level_pct"] == 72.0
    assert values["genset.starting_battery_voltage_v"] == pytest.approx(26.1)
    assert values["genset.output_frequency_hz"] == pytest.approx(50.0)
    assert values["genset.status"] == 3.0


@pytest.mark.asyncio
async def test_thirty_two_bit_values_decode_high_word_first() -> None:
    """DSE is big-endian. Reading it Deye's way turns 230.5 V into 6.5 MV."""
    values = await _poll(_synthetic_dse())

    assert values["genset.output_voltage_v"] == pytest.approx(230.5)
    assert values["genset.energy_kwh"] == pytest.approx(4567.8)
    assert values["genset.starts"] == 1234.0


@pytest.mark.asyncio
async def test_total_power_is_read_as_watts() -> None:
    """Both sources say watts; as kW this is a quarter of a megawatt."""
    values = await _poll(_synthetic_dse())
    assert values["genset.power_w"] == 184000.0


@pytest.mark.asyncio
async def test_run_time_seconds_are_converted_to_hours() -> None:
    values = await _poll(_synthetic_dse())
    assert values["genset.runtime_hours"] == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_load_percent_carries_its_tenth_scale() -> None:
    values = await _poll(_synthetic_dse())
    assert values["genset.load_pct"] == pytest.approx(73.6)


@pytest.mark.asyncio
async def test_a_negative_signed_power_survives_the_sign() -> None:
    """Sets do motor; a reverse-power reading must not read as a huge positive."""
    words = _synthetic_dse()
    words[1536], words[1537] = _u32_be(0x100000000 - 5000)  # -5000 W
    values = await _poll(words)
    assert values["genset.power_w"] == -5000.0


@pytest.mark.asyncio
async def test_the_starts_counter_spans_two_registers() -> None:
    """The spec listed 1808 alone; a 16-bit read wraps past 65535 starts."""
    words = _synthetic_dse()
    words[1808], words[1809] = _u32_be(99999)
    values = await _poll(words)
    assert values["genset.starts"] == 99999.0


# ── The sentinel convention ─────────────────────────────────────


@pytest.mark.asyncio
async def test_an_unimplemented_sensor_publishes_nothing_not_sixty_five_thousand() -> (
    None
):
    """A controller with no oil-pressure sender writes 0xFFFF."""
    words = _synthetic_dse()
    words[1024] = 0xFFFF
    values = await _poll(words)

    assert "genset.oil_pressure_kpa" not in values
    # Its neighbours in the same block are unaffected.
    assert values["genset.coolant_temperature_c"] == 85.0
    assert values["genset.fuel_level_pct"] == 72.0


@pytest.mark.parametrize(
    "sentinel, meaning",
    [
        (0xFFFF, "unimplemented"),
        (0xFFFE, "over range"),
        (0xFFFD, "under range"),
        (0xFFFC, "transducer fault"),
        (0xFFFB, "bad data"),
        (0xFFFA, "high digital input"),
        (0xFFF9, "low digital input"),
        (0xFFF8, "reserved"),
    ],
)
@pytest.mark.asyncio
async def test_every_documented_unsigned_sentinel_declines_to_publish(
    sentinel: int, meaning: str
) -> None:
    words = _synthetic_dse()
    words[1027] = sentinel
    values = await _poll(words)
    assert "genset.fuel_level_pct" not in values, meaning


@pytest.mark.asyncio
async def test_the_signed_sentinel_range_is_declared_too() -> None:
    """Signed registers use 0x7FFF downwards, not 0xFFFF."""
    words = _synthetic_dse()
    words[1025] = 0x7FFF
    values = await _poll(words)
    assert "genset.coolant_temperature_c" not in values

    words[1025] = 0x7FF8
    values = await _poll(words)
    assert "genset.coolant_temperature_c" not in values


@pytest.mark.asyncio
async def test_a_thirty_two_bit_sentinel_declines_to_publish() -> None:
    words = _synthetic_dse()
    words[1800], words[1801] = _u32_be(0xFFFFFFFF)
    values = await _poll(words)
    assert "genset.energy_kwh" not in values


@pytest.mark.asyncio
async def test_a_real_value_just_below_the_sentinel_range_is_published() -> None:
    """0xFFF7 is a reading, not a sentinel. The boundary has to be exact."""
    words = _synthetic_dse()
    words[1024] = 0xFFF7
    values = await _poll(words)
    assert values["genset.oil_pressure_kpa"] == float(0xFFF7)


@pytest.mark.asyncio
async def test_an_unnamed_engine_state_declines_rather_than_emitting_a_code() -> None:
    """8-13 are SAE-reserved and 15 is 'not available'."""
    for code in (9, 14, 15):
        words = _synthetic_dse()
        words[1408] = code
        values = await _poll(words)
        assert "genset.status" not in values, code


# ── Fuel is never inferred ──────────────────────────────────────


def test_the_dse_map_claims_no_fuel_flow() -> None:
    """Not in GenComm; it is ECU dependent and never derived from load %."""
    loaded = load_vendor_map("dse_gencomm")
    metrics = {spec.metric for spec in loaded.all_registers}

    assert "genset.fuel_flow_lph" not in metrics
    assert "genset.fuel_consumed_l" not in metrics
    # The kWh counter is present, and is electrical output only.
    assert "genset.energy_kwh" in metrics


def test_the_map_says_its_kwh_counter_is_not_fuel_evidence() -> None:
    text = open("collector/adapters/vendor_maps/dse_gencomm.yaml").read()
    assert "never on its own evidence of fuel burned" in text


# ── Serial framing and per-site overrides ───────────────────────


def test_the_dse_map_defaults_to_its_documented_serial_settings() -> None:
    loaded = load_vendor_map("dse_gencomm")
    assert loaded.protocol["baudrate"] == 115200
    assert loaded.protocol["unit_id"] == 10
    assert loaded.protocol["parity"] == "N"
    assert loaded.protocol["stopbits"] == 1


def test_a_site_can_override_the_baud_and_slave_the_installer_actually_used() -> None:
    """S1 probes 19200 as well as 115200, and slave 1 as well as 10."""
    resolved = resolve_vendor_map(
        "dse_gencomm",
        {"asset_mappings": dict(ROLES), "baudrate": 19200, "unit_id": 1},
    )
    assert resolved.protocol["baudrate"] == 19200
    assert all(spec.unit_id == 1 for spec in resolved.all_registers)


def test_counters_are_read_on_the_slow_group() -> None:
    loaded = load_vendor_map("dse_gencomm")
    cadence = {group.name: group.poll_interval_seconds for group in loaded.groups}
    assert cadence["counters"] == 300
    assert cadence["engine"] == 30


# ── The ComAp template ──────────────────────────────────────────


def test_the_comap_file_loads_for_inspection() -> None:
    """A technician has to be able to read it before filling it in."""
    loaded = load_vendor_map("comap_intelilite_template")

    assert loaded.vendor == "comap"
    assert loaded.verification == "community"
    assert loaded.complete is False
    assert len(loaded.missing_addresses) >= 10


def test_the_comap_template_refuses_to_poll_until_it_is_filled() -> None:
    with pytest.raises(VendorMapError, match="template"):
        resolve_vendor_map("comap_intelilite_template", {"asset_mappings": dict(ROLES)})


def test_the_template_names_every_metric_it_expects_to_carry() -> None:
    """The roles are the fixed part; only the addresses are per-controller."""
    loaded = load_vendor_map("comap_intelilite_template")
    expected = {
        "start_battery_v",
        "status",
        "oil_pressure",
        "coolant_temp",
        "fuel_level",
        "power_total",
        "output_voltage",
        "frequency",
        "load_pct",
        "runtime",
        "energy",
        "starts",
    }
    assert expected <= set(loaded.missing_addresses)


def test_the_template_carries_comaps_documented_serial_framing() -> None:
    loaded = load_vendor_map("comap_intelilite_template")
    assert loaded.protocol["baudrate"] == 57600
    assert loaded.protocol["parity"] == "N"
    assert loaded.protocol["stopbits"] == 1


def test_the_template_is_configured_for_one_object_per_request() -> None:
    """ComAp answers a single object at a time; block reads are unavailable."""
    loaded = load_vendor_map("comap_intelilite_template")
    assert loaded.protocol["max_gap"] == 0
    assert loaded.protocol["max_block_words"] == 2
    assert loaded.protocol["request_spacing_ms"] >= 5


def test_a_filled_template_polls(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filling the addresses is all it takes — the roles are already right."""
    from pathlib import Path

    from collector.adapters import vendor_map as vm

    source = Path("collector/adapters/vendor_maps/comap_intelilite_template.yaml")
    text = source.read_text()
    # Replace each blank in file order with a plausible exported address.
    exported = [1053, 1000, 1001, 1002, 1003, 1004, 1006, 1007, 1008, 1009, 1011, 1013]
    for address in exported:
        text = text.replace("address: ~", f"address: {address}", 1)
    assert "address: ~" not in text, (
        "the template gained a blank this test does not fill"
    )

    (tmp_path / "comap_site.yaml").write_text(text)
    monkeypatch.setattr(vm, "VENDOR_MAPS_DIR", tmp_path)

    loaded = vm.load_vendor_map("comap_site")
    assert loaded.complete is True
    assert loaded.missing_addresses == ()

    resolved = vm.resolve_vendor_map("comap_site", {"asset_mappings": dict(ROLES)})
    by_id = {spec.id: spec for spec in resolved.all_registers}
    assert by_id["start_battery_v"].address == 1053
    assert by_id["start_battery_v"].asset_id == GENSET
    assert by_id["runtime"].metric == "genset.runtime_hours"
    # Still community, still read-only: filling addresses proves nothing.
    assert resolved.verification == "community"


# ── Provenance and safety ───────────────────────────────────────


@pytest.mark.parametrize("name", ["dse_gencomm", "comap_intelilite_template"])
def test_genset_files_are_community_and_read_only(name: str) -> None:
    loaded = load_vendor_map(name)
    assert loaded.verification == "community"
    assert loaded.source
    assert all(spec.function_code in (3, 4) for spec in loaded.all_registers)


def test_the_dse_map_marks_the_family_specific_register() -> None:
    text = open("collector/adapters/vendor_maps/dse_gencomm.yaml").read()
    assert "UNVERIFIED — bench required" in text
    assert "61xx MkII" in text


def test_the_dse_map_records_the_three_spec_corrections() -> None:
    text = open("collector/adapters/vendor_maps/dse_gencomm.yaml").read()
    assert "WATTS, not kW" in text
    assert "32-bit PAIR" in text
    assert "1558 is scale 0.1" in text


@pytest.mark.asyncio
async def test_a_dse_poll_never_writes() -> None:
    async with _Server(_synthetic_dse()) as server:
        adapter = GenericModbusAdapter()
        await adapter.initialize(
            {
                "vendor_map": "dse_gencomm",
                "protocol": "tcp",
                "host": "127.0.0.1",
                "port": server.port,
                "unit_id": 10,
                "asset_mappings": dict(ROLES),
            }
        )
        await adapter.poll()
        assert adapter.capabilities().can_write is False
        assert await adapter.write("genset.run_command", 1.0) is False
        await adapter.shutdown()
