"""SunSpec float models, storage, MPPT and vendor presets (plan Task 9).

Offsets come from the SunSpec Alliance's own model definitions
(github.com/sunspec/models @ 90b4a331), not from a community transcription,
and the register blocks below are built by hand so the test proves the offset
rather than agreeing with the implementation.

The integer models 101-103 and 201-203 are NOT touched here. Their existing
tests are frozen and green, and the meter import reading the C&I ledger once
called missing is already implemented — this extends the same treatment to
the float meters rather than adding a second import path.
"""

from __future__ import annotations

import struct
from datetime import UTC, datetime

import pytest

from collector.adapters.sunspec_modbus import ModelBlock, SunSpecModbusAdapter
from collector.adapters.vendor_map import VendorMapError, load_sunspec_preset

INVERTER = "60000000-0000-0000-0000-000000000001"
METER = "60000000-0000-0000-0000-000000000002"
BATTERY = "60000000-0000-0000-0000-000000000003"
SOLAR = "60000000-0000-0000-0000-000000000004"


def _f32(value: float) -> tuple[int, int]:
    """SunSpec float32: high word at the LOWER address."""
    return tuple(struct.unpack(">HH", struct.pack(">f", value)))  # type: ignore[return-value]


def _put(registers: list[int], offset: int, value: float) -> None:
    high, low = _f32(value)
    registers[offset], registers[offset + 1] = high, low


NAN_HIGH, NAN_LOW = 0x7FC0, 0x0000


# ── Float inverter models 111/112/113 ───────────────────────────


def _float_inverter_block() -> list[int]:
    registers = [NAN_HIGH if index % 2 == 0 else NAN_LOW for index in range(62)]
    registers[0:2] = [113, 60]
    _put(registers, 2, 12.5)  # A
    _put(registers, 4, 4.1)  # AphA
    _put(registers, 6, 4.2)  # AphB
    _put(registers, 8, 4.3)  # AphC
    _put(registers, 16, 230.5)  # PhVphA
    _put(registers, 18, 231.0)  # PhVphB
    _put(registers, 20, 229.5)  # PhVphC
    _put(registers, 22, 8600.0)  # W
    _put(registers, 24, 50.01)  # Hz
    _put(registers, 40, 42.5)  # TmpCab
    registers[48] = 4  # St
    registers[50:52] = [0x0000, 0x0008]  # Evt1
    return registers


def _decode_float_inverter() -> dict[str, float]:
    adapter = SunSpecModbusAdapter()
    adapter._inverter_asset_id = INVERTER
    readings = adapter._decode_float_inverter(
        _float_inverter_block(), ModelBlock(113, 40070, 60), datetime.now(UTC)
    )
    return {reading.metric_name: reading.value for reading in readings}


def test_float_inverter_decodes_its_mandatory_points() -> None:
    values = _decode_float_inverter()

    assert values["inverter.ac_output_current_a"] == pytest.approx(12.5)
    assert values["inverter.ac_output_voltage_v"] == pytest.approx(230.5)
    assert values["inverter.ac_output_power_w"] == pytest.approx(8600.0)
    assert values["inverter.ac_output_freq_hz"] == pytest.approx(50.01)
    assert values["inverter.temperature_c"] == pytest.approx(42.5)


def test_float_inverter_decodes_all_three_phases() -> None:
    values = _decode_float_inverter()

    assert values["inverter.ac_output_l1_voltage_v"] == pytest.approx(230.5)
    assert values["inverter.ac_output_l2_voltage_v"] == pytest.approx(231.0)
    assert values["inverter.ac_output_l3_voltage_v"] == pytest.approx(229.5)
    assert values["inverter.ac_output_l1_current_a"] == pytest.approx(4.1)
    assert values["inverter.ac_output_l3_current_a"] == pytest.approx(4.3)


def test_float_inverter_state_and_event_stay_integers() -> None:
    """St and Evt1 are enum16/bitfield32 even inside a float model."""
    values = _decode_float_inverter()
    assert values["inverter.state"] == 4.0
    assert values["inverter.error_code"] == 8.0


def test_a_float_point_that_is_nan_is_not_implemented_and_emits_nothing() -> None:
    """NaN survives arithmetic silently; it must never reach a reading."""
    registers = _float_inverter_block()
    registers[22], registers[23] = NAN_HIGH, NAN_LOW  # W not implemented
    adapter = SunSpecModbusAdapter()
    adapter._inverter_asset_id = INVERTER
    readings = adapter._decode_float_inverter(
        registers, ModelBlock(113, 40070, 60), datetime.now(UTC)
    )
    values = {reading.metric_name: reading.value for reading in readings}

    assert "inverter.ac_output_power_w" not in values
    assert values["inverter.ac_output_freq_hz"] == pytest.approx(50.01)


def test_a_short_float_block_decodes_what_is_there_without_raising() -> None:
    registers = _float_inverter_block()[:26]
    adapter = SunSpecModbusAdapter()
    adapter._inverter_asset_id = INVERTER
    readings = adapter._decode_float_inverter(
        registers, ModelBlock(113, 40070, 60), datetime.now(UTC)
    )
    values = {reading.metric_name: reading.value for reading in readings}

    assert values["inverter.ac_output_power_w"] == pytest.approx(8600.0)
    assert "inverter.temperature_c" not in values


def test_float_readings_are_attributed_to_the_inverter_asset() -> None:
    adapter = SunSpecModbusAdapter()
    adapter._inverter_asset_id = INVERTER
    readings = adapter._decode_float_inverter(
        _float_inverter_block(), ModelBlock(113, 40070, 60), datetime.now(UTC)
    )
    assert {reading.asset_id for reading in readings} == {INVERTER}
    assert all(r.raw_register.startswith("SunSpec:113:") for r in readings)


# ── Float meter models 211/212/213 ──────────────────────────────


def _float_meter_block() -> list[int]:
    registers = [NAN_HIGH if index % 2 == 0 else NAN_LOW for index in range(126)]
    registers[0:2] = [213, 124]
    _put(registers, 2, 100.0)  # A
    _put(registers, 4, 33.3)  # AphA
    _put(registers, 6, 33.4)  # AphB
    _put(registers, 8, 33.3)  # AphC
    _put(registers, 10, 230.5)  # PhV
    _put(registers, 12, 230.0)  # PhVphA
    _put(registers, 14, 231.0)  # PhVphB
    _put(registers, 16, 230.5)  # PhVphC
    _put(registers, 26, 50.02)  # Hz
    _put(registers, 28, -2500.0)  # W
    _put(registers, 30, 100.0)  # WphA
    _put(registers, 32, 200.0)  # WphB
    _put(registers, 34, 300.0)  # WphC
    _put(registers, 52, 99.5)  # PF, reported as a percentage
    _put(registers, 60, 100_000.0)  # TotWhExp
    _put(registers, 68, 250_000.0)  # TotWhImp
    return registers


def _decode_float_meter(*, pcc: bool) -> dict[str, float]:
    adapter = SunSpecModbusAdapter()
    adapter._meter_asset_id = METER
    adapter._pcc = pcc
    readings = adapter._decode_float_meter(
        _float_meter_block(), ModelBlock(213, 40138, 124), datetime.now(UTC)
    )
    return {reading.metric_name: reading.value for reading in readings}


def test_a_float_customer_meter_keeps_the_meter_names() -> None:
    values = _decode_float_meter(pcc=False)

    assert values["meter.current_a"] == pytest.approx(100.0)
    assert values["meter.voltage_v"] == pytest.approx(230.5)
    assert values["meter.power_w"] == pytest.approx(-2500.0)
    assert values["meter.power_factor"] == pytest.approx(0.995)
    assert values["meter.energy_kwh"] == pytest.approx(100.0)
    assert not any(name.startswith("grid.") for name in values)


def test_a_float_pcc_meter_reads_both_energy_directions() -> None:
    """The same treatment the integer models already had, not a second path."""
    values = _decode_float_meter(pcc=True)

    assert values["grid.export_energy_kwh"] == pytest.approx(100.0)
    assert values["grid.import_energy_kwh"] == pytest.approx(250.0)
    assert values["grid.voltage_v"] == pytest.approx(230.5)
    assert values["grid.frequency_hz"] == pytest.approx(50.02)
    assert values["system.grid_power_w"] == pytest.approx(-2500.0)
    assert "meter.energy_kwh" not in values
    assert "meter.power_w" not in values


def test_float_meter_phase_detail_decodes() -> None:
    values = _decode_float_meter(pcc=False)

    assert values["meter.l1_voltage_v"] == pytest.approx(230.0)
    assert values["meter.l2_voltage_v"] == pytest.approx(231.0)
    assert values["meter.l2_power_w"] == pytest.approx(200.0)
    assert values["meter.l3_current_a"] == pytest.approx(33.3)


def test_the_float_and_integer_meters_agree_on_the_same_plant() -> None:
    """An installer's float/int+SF choice must not change the numbers."""
    float_values = _decode_float_meter(pcc=True)

    integer = [0] * 107
    integer[0:2] = [203, 105]
    integer[7], integer[15] = 2305, 0xFFFF  # 230.5 V
    integer[18], integer[22] = 0xF63C, 0  # -2500 W
    integer[16], integer[17] = 5002, 0xFFFE  # 50.02 Hz
    integer[38:40] = [0x0001, 0x86A0]  # 100,000 Wh
    integer[46:48] = [0x0003, 0xD090]  # 250,000 Wh
    integer[54] = 0
    adapter = SunSpecModbusAdapter()
    adapter._meter_asset_id = METER
    adapter._pcc = True
    integer_values = {
        reading.metric_name: reading.value
        for reading in adapter._decode_meter(
            integer, ModelBlock(203, 40138, 105), datetime.now(UTC)
        )
    }

    for metric in (
        "grid.voltage_v",
        "grid.frequency_hz",
        "system.grid_power_w",
        "grid.import_energy_kwh",
        "grid.export_energy_kwh",
    ):
        assert float_values[metric] == pytest.approx(integer_values[metric]), metric


# ── Model 124, storage — read only ──────────────────────────────


def _storage_block(soc_raw: int = 875, scale: int = 0xFFFF) -> list[int]:
    registers = [0] * 26
    registers[0:2] = [124, 24]
    registers[2] = 5000  # WChaMax — settable, deliberately not emitted
    registers[5] = 0x0003  # StorCtl_Mod
    registers[8] = soc_raw
    registers[18] = 0  # WChaMax_SF
    registers[22] = scale  # ChaState_SF = -1
    return registers


def test_storage_reports_state_of_charge() -> None:
    adapter = SunSpecModbusAdapter()
    adapter._battery_asset_id = BATTERY
    readings = adapter._decode_storage(
        _storage_block(), ModelBlock(124, 40200, 24), datetime.now(UTC)
    )
    values = {reading.metric_name: reading.value for reading in readings}

    assert values["battery.soc_pct"] == pytest.approx(87.5)
    assert readings[0].asset_id == BATTERY


def test_storage_emits_nothing_else_from_the_model() -> None:
    """WChaMax and StorCtl_Mod are setpoints, and this adapter never writes.

    battery.max_charge_w is a WRITE metric deliberately excluded from
    ALL_METRICS, so publishing WChaMax as telemetry would be rejected at
    ingest anyway — and a setpoint reported as a measurement is its own kind
    of wrong.
    """
    adapter = SunSpecModbusAdapter()
    adapter._battery_asset_id = BATTERY
    readings = adapter._decode_storage(
        _storage_block(), ModelBlock(124, 40200, 24), datetime.now(UTC)
    )
    assert {reading.metric_name for reading in readings} == {"battery.soc_pct"}


def test_storage_without_a_battery_asset_emits_nothing() -> None:
    """Ingest rejects a battery metric addressed to an inverter asset."""
    adapter = SunSpecModbusAdapter()
    adapter._inverter_asset_id = INVERTER  # but no battery bound
    readings = adapter._decode_storage(
        _storage_block(), ModelBlock(124, 40200, 24), datetime.now(UTC)
    )
    assert readings == []


def test_storage_honours_the_not_implemented_sentinel() -> None:
    adapter = SunSpecModbusAdapter()
    adapter._battery_asset_id = BATTERY
    readings = adapter._decode_storage(
        _storage_block(soc_raw=0xFFFF), ModelBlock(124, 40200, 24), datetime.now(UTC)
    )
    assert readings == []


# ── Model 160, multiple MPPT ────────────────────────────────────


def _mppt_block(modules: int = 2) -> list[int]:
    registers = [0] * (10 + 20 * modules)
    registers[0:2] = [160, 8 + 20 * modules]
    registers[2] = 0xFFFF  # DCA_SF = -1
    registers[3] = 0xFFFF  # DCV_SF = -1
    registers[4] = 0  # DCW_SF = 0
    registers[8] = modules  # N
    for index in range(modules):
        base = 10 + 20 * index
        registers[base] = index + 1
        registers[base + 9] = 79 + index  # DCA -> 7.9, 8.0
        registers[base + 10] = 3800 + index  # DCV -> 380.0, 380.1
        registers[base + 11] = 3000 + index * 500  # DCW
    return registers


def _decode_mppt(modules: int = 2) -> dict[str, float]:
    adapter = SunSpecModbusAdapter()
    adapter._pv_asset_id = SOLAR
    readings = adapter._decode_mppt(
        _mppt_block(modules),
        ModelBlock(160, 40300, 8 + 20 * modules),
        datetime.now(UTC),
    )
    return {reading.metric_name: reading.value for reading in readings}


def test_each_mppt_module_reports_current_voltage_and_power() -> None:
    values = _decode_mppt()

    assert values["solar.mppt1_current_a"] == pytest.approx(7.9)
    assert values["solar.mppt1_voltage_v"] == pytest.approx(380.0)
    assert values["solar.mppt1_power_w"] == pytest.approx(3000.0)
    assert values["solar.mppt2_current_a"] == pytest.approx(8.0)
    assert values["solar.mppt2_power_w"] == pytest.approx(3500.0)


def test_the_module_stride_is_twenty_registers() -> None:
    """An 18- or 22-register stride decodes cleanly and reports nonsense."""
    values = _decode_mppt(modules=3)
    assert values["solar.mppt3_power_w"] == pytest.approx(4000.0)
    assert values["solar.mppt3_voltage_v"] == pytest.approx(380.2)


def test_scale_factors_are_shared_across_every_module() -> None:
    """One set for the whole model, not one per module."""
    values = _decode_mppt(modules=3)
    for index in (1, 2, 3):
        assert values[f"solar.mppt{index}_voltage_v"] > 100  # -1 scale applied


def test_a_module_count_beyond_what_was_read_is_bounded() -> None:
    """A wrong N must not walk off the end of the block."""
    registers = _mppt_block(modules=2)
    registers[8] = 40  # claims forty modules
    adapter = SunSpecModbusAdapter()
    adapter._pv_asset_id = SOLAR
    readings = adapter._decode_mppt(
        registers, ModelBlock(160, 40300, 48), datetime.now(UTC)
    )
    modules = {reading.metric_name for reading in readings}
    assert "solar.mppt2_power_w" in modules
    assert "solar.mppt3_power_w" not in modules


def test_module_count_is_capped_at_the_catalogues_range() -> None:
    """solar.mppt{n}_* exists for n in 1..8; a ninth has no name to use."""
    values = _decode_mppt(modules=10)
    assert "solar.mppt8_power_w" in values
    assert "solar.mppt9_power_w" not in values


def test_mppt_readings_go_to_the_pv_asset() -> None:
    adapter = SunSpecModbusAdapter()
    adapter._pv_asset_id = SOLAR
    adapter._inverter_asset_id = INVERTER
    readings = adapter._decode_mppt(
        _mppt_block(), ModelBlock(160, 40300, 48), datetime.now(UTC)
    )
    assert {reading.asset_id for reading in readings} == {SOLAR}


def test_mppt_falls_back_to_the_inverter_asset_when_no_pv_asset_is_bound() -> None:
    """The catalogue admits solar.* on a charge_controller or solar_array; an
    unbound PV asset would otherwise silently drop every string."""
    adapter = SunSpecModbusAdapter()
    adapter._inverter_asset_id = INVERTER
    readings = adapter._decode_mppt(
        _mppt_block(), ModelBlock(160, 40300, 48), datetime.now(UTC)
    )
    assert {reading.asset_id for reading in readings} == {INVERTER}


# ── Vendor presets ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["sunspec_sma", "sunspec_solaredge", "sunspec_fronius"]
)
def test_every_preset_declares_its_tier_and_source(name: str) -> None:
    preset = load_sunspec_preset(name)
    assert preset["verification"] == "community"
    assert preset["source"].strip()
    assert preset["adapter"] == "sunspec_modbus"


def test_the_sma_preset_carries_its_unusual_unit_id() -> None:
    preset = load_sunspec_preset("sunspec_sma")
    assert preset["connection"]["unit_id"] == 126
    assert preset["connection"]["port"] == 502


def test_the_solaredge_preset_carries_its_unusual_port() -> None:
    """1502, not 502 — the most common misconfiguration on this brand."""
    preset = load_sunspec_preset("sunspec_solaredge")
    assert preset["connection"]["port"] == 1502
    assert preset["connection"]["unit_id"] == 1
    assert "idle timeout" in preset["notes"]


def test_the_fronius_preset_records_both_meter_unit_ids() -> None:
    preset = load_sunspec_preset("sunspec_fronius")
    assert preset["meter"]["gen24_unit_id"] == 200
    assert preset["meter"]["datamanager_unit_id"] == 240


@pytest.mark.parametrize(
    "name, excluded",
    [
        ("sunspec_sma", "Sunny Island"),
        ("sunspec_solaredge", "STORAGE"),
    ],
)
def test_presets_state_what_they_cannot_read(name: str, excluded: str) -> None:
    """An exclusion stated is a support call that does not happen."""
    assert excluded in load_sunspec_preset(name)["exclusions"]


def test_a_preset_is_not_loadable_as_a_register_map() -> None:
    from collector.adapters.vendor_map import load_vendor_map

    with pytest.raises(VendorMapError):
        load_vendor_map("sunspec_sma")


def test_a_register_map_is_not_loadable_as_a_preset() -> None:
    with pytest.raises(VendorMapError, match="not a SunSpec preset"):
        load_sunspec_preset("deye_sg04lp3")


def test_presets_declare_no_write_surface() -> None:
    for name in ("sunspec_sma", "sunspec_solaredge", "sunspec_fronius"):
        preset = load_sunspec_preset(name)
        assert "write_mappings" not in preset
        assert "write" not in preset["connection"]


# ── The adapter still refuses to write ──────────────────────────


@pytest.mark.asyncio
async def test_the_sunspec_adapter_declares_no_write_capability() -> None:
    adapter = SunSpecModbusAdapter()
    caps = adapter.capabilities()
    assert caps.can_write is False
    assert await adapter.write("battery.soc_pct", 50.0) is False


def test_capabilities_now_advertise_storage_and_mppt_metrics() -> None:
    metrics = set(SunSpecModbusAdapter().capabilities().supported_metrics)
    assert "battery.soc_pct" in metrics
    assert "solar.mppt1_power_w" in metrics
    assert "solar.mppt8_current_a" in metrics


# ── The --sunspec probe mode ────────────────────────────────────

from typing import Any  # noqa: E402

from collector.register_probe import _build_arg_parser, _sunspec_scale_factors  # noqa: E402


def test_the_probe_cli_offers_sunspec_as_an_alternative_to_a_map() -> None:
    args = _build_arg_parser().parse_args(["--sunspec", "--host", "10.0.0.5"])
    assert args.sunspec is True
    assert args.vendor_map is None


def test_the_probe_cli_refuses_both_modes_at_once() -> None:
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["--sunspec", "--vendor-map", "deye_sg04lp3"])


def test_the_probe_cli_still_requires_one_mode() -> None:
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["--host", "10.0.0.5"])


def test_the_probe_cli_accepts_a_preset() -> None:
    args = _build_arg_parser().parse_args(
        ["--sunspec", "--preset", "sunspec_solaredge", "--host", "10.0.0.5"]
    )
    assert args.preset == "sunspec_solaredge"


def test_scale_factors_skip_the_not_implemented_sentinel() -> None:
    """0x8000 is sunssf's "not implemented", not a scale of -32768."""
    factors = _sunspec_scale_factors([0x8000, 0xFFFF, 0x0002, 0x1234])
    assert "offset_0" not in factors
    assert factors["offset_1"] == -1
    assert factors["offset_2"] == 2
    # 0x1234 is 4660 — far outside the plausible sunssf range.
    assert "offset_3" not in factors


@pytest.mark.asyncio
async def test_a_sunspec_sweep_records_the_model_chain(edge_db: Any) -> None:
    from unittest.mock import AsyncMock, patch

    from collector.register_probe import SunSpecProbe

    probe = SunSpecProbe(
        db=edge_db,
        transport={"protocol": "tcp", "host": "127.0.0.1", "port": 502},
        preset="sunspec_sma",
    )
    with patch(
        "collector.adapters.sunspec_modbus.SunSpecModbusAdapter.initialize",
        AsyncMock(return_value=False),
    ):
        record = await probe.run_once(now_ms=1)

    # A device that will not answer still leaves a record saying so.
    assert record["connected"] is False
    assert record["verification"] == "community"
    assert "does not promote" in record["verification_note"]
    stored = edge_db.get_vendor_map_validations(map_name="sunspec_sma")
    assert len(stored) == 1

    items = edge_db.get_queue_items(priority=3, limit=10)
    assert len(items) == 1
    assert items[0]["payload"]["type"] == "edge.sunspec_validation"
    assert items[0]["payload"]["detail"].endswith("vendor_map_validations")


@pytest.mark.asyncio
async def test_a_sunspec_sweep_keeps_the_operator_blanks(edge_db: Any) -> None:
    from unittest.mock import AsyncMock, patch

    from collector.register_probe import SunSpecProbe

    probe = SunSpecProbe(
        db=edge_db, transport={"protocol": "tcp", "host": "127.0.0.1"}, preset=None
    )
    with patch(
        "collector.adapters.sunspec_modbus.SunSpecModbusAdapter.initialize",
        AsyncMock(return_value=False),
    ):
        record = await probe.run_once(now_ms=1)

    # The installer's float-vs-int+SF choice changes which offsets were
    # exercised, so the record has somewhere to say which was selected.
    assert "model_type_selected" in record["operator"]
    assert all(value is None for value in record["operator"].values())
    assert set(record["pending_checks"]) == {
        "counter_monotonicity",
        "sign_conventions",
        "display_comparison",
    }
