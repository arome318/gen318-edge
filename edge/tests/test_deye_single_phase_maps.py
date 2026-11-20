"""Deye single-phase hybrid maps against a real Modbus server.

Same method as test_deye_vendor_maps.py: a real pymodbus TCP server on
loopback, seeded with a SYNTHETIC register image, polled through the
production adapter. Nothing here is a captured dump from a physical inverter;
no such dump exists yet, and the maps say so (verification: community).

The values are chosen so that each likely transcription error produces an
obviously wrong number rather than a plausible one: a missed sign inversion,
a 10x scale, adjacent-word decoding of the split grid-import counter, and the
three-phase map's address range applied to a single-phase unit.
"""

from __future__ import annotations

from typing import Any

import pytest

from collector.adapters.generic_modbus import GenericModbusAdapter
from collector.adapters.vendor_map import (
    VendorMapError,
    load_vendor_map,
    resolve_vendor_map,
)
from tests.test_deye_vendor_maps import _Server

INVERTER = "50000000-0000-0000-0000-000000000001"
BATTERY = "50000000-0000-0000-0000-000000000002"
SOLAR = "50000000-0000-0000-0000-000000000003"
GRID = "50000000-0000-0000-0000-000000000004"
LOAD = "50000000-0000-0000-0000-000000000005"
GENSET = "50000000-0000-0000-0000-000000000006"

ROLES = {
    "inverter": INVERTER,
    "battery": BATTERY,
    "solar_array": SOLAR,
    "grid_connection": GRID,
}

SINGLE_PHASE_MAPS = ("deye_sg03lp1", "deye_sg02lp1")


def _u32_le(value: int) -> tuple[int, int]:
    return value & 0xFFFF, (value >> 16) & 0xFFFF


def _i16(value: int) -> int:
    """A signed value as the unsigned word the inverter puts on the wire."""
    return value & 0xFFFF


def _synthetic_single_phase(*, rated_w: int = 8000) -> list[int]:
    """A plausible single-phase hybrid register image. SYNTHETIC.

    The battery is DISCHARGING 1500 W and the site is IMPORTING 1200 W, in
    Deye's own wire convention (positive = battery output, positive = buy).
    """
    words = [0] * 400

    # Identity: device type 0x0003 (single-phase hybrid), protocol 1.4,
    # serial "2410A" in five ASCII registers, rated power in 0.1 W.
    words[0] = 0x0003
    words[2] = 0x0104
    words[3:8] = [0x3234, 0x3130, 0x4100, 0x0000, 0x0000]
    words[16], words[17] = _u32_le(rated_w * 10)

    words[59] = 2  # normal
    words[90] = 1500  # DC transformer 50.0 degC (identity/evidence only)
    words[91] = 1450  # IGBT / AC radiator 45.0 degC
    words[103:107] = [0, 0, 0, 0]  # no faults

    # Energy counters. Lifetime import is the split one: low word at 78,
    # grid frequency at 79, high word at 80.
    words[72], words[73] = _u32_le(54321)  # 5432.1 kWh charged
    words[74], words[75] = _u32_le(45678)  # 4567.8 kWh discharged
    low, high = _u32_le(98765)  # 9876.5 kWh imported, high word = 1
    words[78], words[80] = low, high
    words[79] = 5002  # 50.02 Hz — NOT part of the import counter
    words[81], words[82] = _u32_le(11111)  # 1111.1 kWh exported
    words[96], words[97] = _u32_le(123456)  # 12345.6 kWh PV lifetime
    words[108] = 421  # 42.1 kWh PV today

    # PV: MPPT1 3000 W at 380.0 V / 7.9 A, MPPT2 2000 W at 375.0 V / 5.3 A
    words[109], words[110] = 3800, 79
    words[111], words[112] = 3750, 53
    words[186], words[187] = 3000, 2000

    # Grid 230.5 V, importing 1200 W
    words[150] = 2305
    words[169] = 1200

    # Inverter output 5000 W at 230.0 V / 21.74 A, 50.01 Hz
    words[154] = 2300
    words[164] = 2174
    words[175] = 5000
    words[193] = 5001

    # Load port 3600 W; gen port 4000 W at 231.0 V / 50.03 Hz
    words[178] = 3600
    words[166] = 4000
    words[181] = 2310
    words[196] = 5003

    # Battery: 31.5 degC, 51.20 V, 88 %, discharging 1500 W / 29.30 A
    words[182] = 1315
    words[183] = 5120
    words[184] = 88
    words[190] = 1500
    words[191] = 2930
    return words


async def _poll(
    map_name: str, words: list[int], **config: Any
) -> tuple[dict[str, float], list[Any]]:
    async with _Server(words) as server:
        adapter = GenericModbusAdapter()
        base: dict[str, Any] = {
            "vendor_map": map_name,
            "protocol": "tcp",
            "host": "127.0.0.1",
            "port": server.port,
            "asset_mappings": dict(ROLES),
        }
        base.update(config)
        assert await adapter.initialize(base) is True, "adapter failed to connect"
        readings = await adapter.poll()
        await adapter.shutdown()
    return {r.metric_name: r.value for r in readings}, readings


async def _identity(map_name: str, words: list[int]) -> dict[str, str]:
    async with _Server(words) as server:
        adapter = GenericModbusAdapter()
        await adapter.initialize(
            {
                "vendor_map": map_name,
                "protocol": "tcp",
                "host": "127.0.0.1",
                "port": server.port,
                "asset_mappings": dict(ROLES),
            }
        )
        identity = adapter.identity()
        await adapter.shutdown()
    return identity


# ── The shipped files ───────────────────────────────────────────


@pytest.mark.parametrize("name", SINGLE_PHASE_MAPS)
def test_single_phase_maps_are_community_read_only_and_rtu(name: str) -> None:
    loaded = load_vendor_map(name)
    assert loaded.vendor == "deye"
    assert loaded.verification == "community"
    assert loaded.source
    assert loaded.protocol["protocol"] == "rtu"
    assert loaded.protocol["baudrate"] == 9600
    assert loaded.protocol["unit_id"] == 1
    assert all(spec.function_code == 3 for spec in loaded.all_registers)


@pytest.mark.parametrize("name", SINGLE_PHASE_MAPS)
def test_single_phase_maps_never_touch_the_three_phase_range(name: str) -> None:
    """The three-phase map lives at 500-700; nothing here may stray there."""
    loaded = load_vendor_map(name)
    for spec in loaded.all_registers + loaded.identity:
        if spec.derived is None:
            assert spec.address + spec.word_count <= 250, spec.id


@pytest.mark.parametrize("name", SINGLE_PHASE_MAPS)
def test_unresolved_questions_are_marked_where_they_are_read(name: str) -> None:
    text = open(f"collector/adapters/vendor_maps/{name}.yaml").read()
    assert "UNVERIFIED — bench required" in text


def test_the_model_families_name_the_models_they_cover() -> None:
    small = load_vendor_map("deye_sg03lp1").model_family
    large = load_vendor_map("deye_sg02lp1").model_family
    for model in ("SG03LP1", "SG05LP1", "SG01LP1"):
        assert model in small
    for model in ("SG02LP1", "SG01LP1", "12K"):
        assert model in large


def test_the_large_map_inherits_the_small_one() -> None:
    assert load_vendor_map("deye_sg02lp1").extends == ("deye_sg03lp1",)


# ── Decoding a synthetic frame, deye_sg03lp1 ────────────────────


@pytest.mark.asyncio
async def test_one_device_lands_on_four_assets() -> None:
    values, readings = await _poll("deye_sg03lp1", _synthetic_single_phase())
    assert {r.asset_id for r in readings} == {INVERTER, BATTERY, SOLAR, GRID}
    by_metric = {r.metric_name: r for r in readings}
    assert by_metric["battery.soc_pct"].asset_id == BATTERY
    assert by_metric["solar.mppt1_power_w"].asset_id == SOLAR
    assert by_metric["grid.frequency_hz"].asset_id == GRID
    assert by_metric["inverter.ac_output_power_w"].asset_id == INVERTER
    assert by_metric["battery.soc_pct"].raw_register == "modbus:1:184"


@pytest.mark.asyncio
async def test_scaling_matches_the_sources() -> None:
    values, _ = await _poll("deye_sg03lp1", _synthetic_single_phase())

    assert values["battery.soc_pct"] == 88.0
    assert values["battery.voltage_v"] == pytest.approx(51.20)  # x0.01
    assert values["grid.voltage_v"] == pytest.approx(230.5)
    assert values["grid.frequency_hz"] == pytest.approx(50.02)
    assert values["inverter.ac_output_power_w"] == 5000.0
    assert values["inverter.ac_output_voltage_v"] == pytest.approx(230.0)
    assert values["inverter.ac_output_current_a"] == pytest.approx(21.74)
    assert values["inverter.ac_output_freq_hz"] == pytest.approx(50.01)
    assert values["solar.mppt1_power_w"] == 3000.0
    assert values["solar.mppt2_power_w"] == 2000.0
    assert values["solar.pv_power_w"] == 5000.0
    assert values["solar.mppt1_voltage_v"] == pytest.approx(380.0)
    assert values["solar.mppt1_current_a"] == pytest.approx(7.9)
    assert values["solar.mppt2_voltage_v"] == pytest.approx(375.0)
    assert values["solar.mppt2_current_a"] == pytest.approx(5.3)
    assert values["inverter.state"] == 2.0
    assert values["inverter.error_code"] == 0.0


@pytest.mark.asyncio
async def test_temperatures_apply_scale_then_offset() -> None:
    values, _ = await _poll("deye_sg03lp1", _synthetic_single_phase())
    assert values["inverter.temperature_c"] == pytest.approx(45.0)  # register 91
    assert values["battery.temperature_c"] == pytest.approx(31.5)


# ── Battery sign: Deye reports OUTPUT, the catalogue wants CHARGING ─


@pytest.mark.asyncio
async def test_a_discharging_battery_reads_negative() -> None:
    """Deye's +1500 W 'output' is a discharge: canonical positive = charging."""
    values, _ = await _poll("deye_sg03lp1", _synthetic_single_phase())
    assert values["battery.power_w"] == -1500.0
    assert values["battery.current_a"] == pytest.approx(-29.30)


@pytest.mark.asyncio
async def test_a_charging_battery_reads_positive() -> None:
    words = _synthetic_single_phase()
    words[190] = _i16(-2500)  # 0xF63C on the wire: charging 2500 W
    words[191] = _i16(-4880)  # charging 48.80 A
    values, _ = await _poll("deye_sg03lp1", words)
    assert values["battery.power_w"] == 2500.0
    assert values["battery.current_a"] == pytest.approx(48.80)


# ── Grid: one signed register, two directions ───────────────────


@pytest.mark.asyncio
async def test_a_positive_grid_reading_is_import_and_export_is_a_real_zero() -> None:
    values, _ = await _poll("deye_sg03lp1", _synthetic_single_phase())
    assert values["grid.import_power_w"] == 1200.0
    assert values["grid.export_power_w"] == 0.0


@pytest.mark.asyncio
async def test_a_negative_grid_reading_becomes_export_as_a_magnitude() -> None:
    words = _synthetic_single_phase()
    words[169] = _i16(-1800)  # 0xF8F8: selling 1800 W
    values, _ = await _poll("deye_sg03lp1", words)
    assert values["grid.import_power_w"] == 0.0
    assert values["grid.export_power_w"] == 1800.0


# ── Lifetime counters ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_lifetime_counters_decode_low_word_first() -> None:
    values, _ = await _poll("deye_sg03lp1", _synthetic_single_phase())
    assert values["solar.total_yield_kwh"] == pytest.approx(12345.6)
    assert values["battery.charge_energy_kwh"] == pytest.approx(5432.1)
    assert values["battery.discharge_energy_kwh"] == pytest.approx(4567.8)
    assert values["grid.export_energy_kwh"] == pytest.approx(1111.1)
    assert values["solar.daily_yield_kwh"] == pytest.approx(42.1)


@pytest.mark.asyncio
async def test_grid_import_takes_its_high_word_from_80_not_from_frequency() -> None:
    """Adjacent-word decoding would report 5002*6553.6 + ... ≈ 32.8 million kWh."""
    values, readings = await _poll("deye_sg03lp1", _synthetic_single_phase())
    assert values["grid.import_energy_kwh"] == pytest.approx(9876.5)
    imp = next(r for r in readings if r.metric_name == "grid.import_energy_kwh")
    assert imp.raw_register == "modbus:1:78"

    # Grid frequency moving must not move the counter.
    words = _synthetic_single_phase()
    words[79] = 4985
    moved, _ = await _poll("deye_sg03lp1", words)
    assert moved["grid.import_energy_kwh"] == pytest.approx(9876.5)
    assert moved["grid.frequency_hz"] == pytest.approx(49.85)


# ── Optional roles ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_port_appears_only_when_a_load_asset_is_bound() -> None:
    without, _ = await _poll("deye_sg03lp1", _synthetic_single_phase())
    assert "load.power_w" not in without

    with_load, readings = await _poll(
        "deye_sg03lp1",
        _synthetic_single_phase(),
        asset_mappings={**ROLES, "load": LOAD},
    )
    assert with_load["load.power_w"] == 3600.0
    assert {r.asset_id for r in readings if r.metric_name == "load.power_w"} == {LOAD}


@pytest.mark.asyncio
async def test_the_gen_port_stays_off_until_bound_and_enabled() -> None:
    bound_only, _ = await _poll(
        "deye_sg03lp1",
        _synthetic_single_phase(),
        asset_mappings={**ROLES, "diesel_genset": GENSET},
    )
    assert not any(metric.startswith("genset.") for metric in bound_only)

    enabled, _ = await _poll(
        "deye_sg03lp1",
        _synthetic_single_phase(),
        asset_mappings={**ROLES, "diesel_genset": GENSET},
        features={"gen_port": True},
    )
    assert enabled["genset.power_w"] == 4000.0
    assert enabled["genset.output_voltage_v"] == pytest.approx(231.0)
    assert enabled["genset.output_frequency_hz"] == pytest.approx(50.03)


@pytest.mark.asyncio
async def test_a_negative_gen_port_reading_is_kept_and_flagged_suspect() -> None:
    """Read signed so an inverted port shows -3000 W, never 62,536 W."""
    words = _synthetic_single_phase()
    words[166] = _i16(-3000)
    _, readings = await _poll(
        "deye_sg03lp1",
        words,
        asset_mappings={**ROLES, "diesel_genset": GENSET},
        features={"gen_port": True},
    )
    gen = next(r for r in readings if r.metric_name == "genset.power_w")
    assert gen.value == -3000.0
    assert gen.quality == 1


# ── Identity, refusals and faults ───────────────────────────────


@pytest.mark.asyncio
async def test_identity_reports_type_serial_and_rated_power_not_telemetry() -> None:
    identity = await _identity("deye_sg03lp1", _synthetic_single_phase(rated_w=8000))
    assert identity["device_type"] == "3.0"
    assert identity["serial"] == "2410A"
    assert identity["rated_power_w"] == "8000.0"
    assert identity["dc_transformer_temp_c"] == "50.0"  # register 90, evidence only

    _, readings = await _poll("deye_sg03lp1", _synthetic_single_phase())
    assert not any(r.metric_name.startswith("identity.") for r in readings)
    assert len([r for r in readings if r.metric_name == "inverter.temperature_c"]) == 1


@pytest.mark.asyncio
async def test_an_undocumented_run_state_emits_nothing() -> None:
    words = _synthetic_single_phase()
    words[59] = 5  # neither C nor D lists a code 5 on single-phase
    values, _ = await _poll("deye_sg03lp1", words)
    assert "inverter.state" not in values


@pytest.mark.asyncio
async def test_a_fault_bit_reports_its_position_and_the_raw_words() -> None:
    words = _synthetic_single_phase()
    words[104] = 0x0004  # bit 2 of the second word -> position 19
    _, readings = await _poll("deye_sg03lp1", words)
    fault = next(r for r in readings if r.metric_name == "inverter.error_code")
    assert fault.value == 19.0
    assert "0000,0004,0000,0000" in fault.raw_register


@pytest.mark.asyncio
async def test_an_out_of_range_soc_is_kept_and_flagged() -> None:
    words = _synthetic_single_phase()
    words[184] = 150
    _, readings = await _poll("deye_sg03lp1", words)
    soc = next(r for r in readings if r.metric_name == "battery.soc_pct")
    assert soc.value == 150.0
    assert soc.quality == 1


# ── deye_sg02lp1: the disputed scales ship OFF ──────────────────


def _large_frame() -> list[int]:
    words = _synthetic_single_phase(rated_w=12000)
    words[188], words[113], words[114] = 1500, 3700, 41  # PV3: 1500 W, 370.0 V, 4.1 A
    words[169] = 120  # 1200 W if the unit counts in 10 W steps
    words[178] = 360  # 3600 W in 10 W steps
    words[191] = 293  # 29.3 A in 0.1 A steps, discharging
    return words


@pytest.mark.asyncio
async def test_the_large_map_emits_no_disputed_reading_until_commissioned() -> None:
    values, _ = await _poll(
        "deye_sg02lp1", _large_frame(), asset_mappings={**ROLES, "load": LOAD}
    )
    for disputed in (
        "grid.import_power_w",
        "grid.export_power_w",
        "load.power_w",
        "battery.current_a",
    ):
        assert disputed not in values, disputed
    # Everything the sources agree on still arrives.
    assert values["battery.power_w"] == -1500.0
    assert values["inverter.ac_output_power_w"] == 5000.0
    assert values["grid.import_energy_kwh"] == pytest.approx(9876.5)


@pytest.mark.asyncio
async def test_the_ten_watt_scale_applies_to_grid_and_load_together() -> None:
    values, _ = await _poll(
        "deye_sg02lp1",
        _large_frame(),
        asset_mappings={**ROLES, "load": LOAD},
        features={"power_scale_10w": True},
    )
    assert values["grid.import_power_w"] == 1200.0
    assert values["grid.export_power_w"] == 0.0
    assert values["load.power_w"] == 3600.0


@pytest.mark.asyncio
async def test_the_one_watt_scale_reads_the_same_words_ten_times_smaller() -> None:
    values, _ = await _poll(
        "deye_sg02lp1", _large_frame(), features={"power_scale_1w": True}
    )
    assert values["grid.import_power_w"] == 120.0


@pytest.mark.asyncio
async def test_ten_watt_export_still_splits_into_a_magnitude() -> None:
    words = _large_frame()
    words[169] = _i16(-180)  # selling 1800 W in 10 W steps
    values, _ = await _poll("deye_sg02lp1", words, features={"power_scale_10w": True})
    assert values["grid.import_power_w"] == 0.0
    assert values["grid.export_power_w"] == 1800.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "feature, raw, expected",
    [
        ("battery_current_0_01a", 2930, -29.30),
        ("battery_current_0_1a", 293, -29.30),
        ("battery_current_0_1a", _i16(-488), 48.80),  # charging
    ],
)
async def test_battery_current_scale_is_chosen_at_commissioning(
    feature: str, raw: int, expected: float
) -> None:
    words = _large_frame()
    words[191] = raw
    values, _ = await _poll("deye_sg02lp1", words, features={feature: True})
    assert values["battery.current_a"] == pytest.approx(expected)


@pytest.mark.asyncio
async def test_the_third_mppt_joins_the_plant_total() -> None:
    values, _ = await _poll("deye_sg02lp1", _large_frame())
    assert values["solar.mppt3_power_w"] == 1500.0
    assert values["solar.mppt3_voltage_v"] == pytest.approx(370.0)
    assert values["solar.mppt3_current_a"] == pytest.approx(4.1)
    assert values["solar.pv_power_w"] == 6500.0


def test_enabling_both_scales_of_one_pair_is_refused() -> None:
    for pair in (
        ("power_scale_1w", "power_scale_10w"),
        ("battery_current_0_01a", "battery_current_0_1a"),
    ):
        with pytest.raises(VendorMapError, match="both emit"):
            resolve_vendor_map(
                "deye_sg02lp1",
                {
                    "asset_mappings": {**ROLES, "load": LOAD},
                    "features": {pair[0]: True, pair[1]: True},
                },
            )


@pytest.mark.asyncio
async def test_a_double_enabled_site_fails_to_start_rather_than_drop_rows() -> None:
    async with _Server(_large_frame()) as server:
        adapter = GenericModbusAdapter()
        started = await adapter.initialize(
            {
                "vendor_map": "deye_sg02lp1",
                "protocol": "tcp",
                "host": "127.0.0.1",
                "port": server.port,
                "asset_mappings": dict(ROLES),
                "features": {"power_scale_1w": True, "power_scale_10w": True},
            }
        )
        await adapter.shutdown()
    assert started is False


@pytest.mark.asyncio
async def test_grid_and_load_can_take_different_scales_when_a_unit_needs_it() -> None:
    """The onboarding recipe for A's 16 kW finding: grid x10, load x1."""
    words = _large_frame()
    words[178] = 3600  # this unit reports load in 1 W
    values, _ = await _poll(
        "deye_sg02lp1",
        words,
        asset_mappings={**ROLES, "load": LOAD},
        features={"power_scale_1w": True, "power_scale_10w": True},
        registers_disable=["grid_w", "load_w_x10"],
    )
    assert values["grid.import_power_w"] == 1200.0
    assert values["load.power_w"] == 3600.0


def test_the_probe_reads_both_candidate_scales_side_by_side() -> None:
    """Commissioning compares the two decodings with the inverter's display."""
    probe = resolve_vendor_map("deye_sg02lp1", {}, require_bindings=False)
    ids = {spec.id for spec in probe.all_registers}
    assert {"grid_w", "grid_w_x10", "load_w", "load_w_x10", "bat_i", "bat_i_x10"} <= ids


# ── Read-only ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_single_phase_poll_never_writes() -> None:
    async with _Server(_synthetic_single_phase()) as server:
        adapter = GenericModbusAdapter()
        await adapter.initialize(
            {
                "vendor_map": "deye_sg03lp1",
                "protocol": "tcp",
                "host": "127.0.0.1",
                "port": server.port,
                "asset_mappings": dict(ROLES),
            }
        )
        await adapter.poll()
        assert adapter.capabilities().can_write is False
        assert await adapter.write("battery.soc_pct", 1.0) is False
        await adapter.shutdown()
