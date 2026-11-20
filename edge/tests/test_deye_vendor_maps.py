"""Deye and SunSynk maps against a real Modbus server (plan Task 7).

These tests stand up an actual pymodbus TCP server on loopback, seed its
registers with a SYNTHETIC dataset, and poll it through the production
adapter. Nothing here is a captured dump from a physical inverter — no such
dump exists, and a synthetic fixture that pretended otherwise would be the
exact dishonesty the spec's verification tiers are for.

A real server is worth the cost: it exercises the pymodbus call signature,
the block planner's request boundaries and the decoders together. A mocked
client agrees with whatever the adapter does, including being wrong.
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import pytest

from collector.adapters.generic_modbus import GenericModbusAdapter
from collector.adapters.vendor_map import load_vendor_map, resolve_vendor_map

INVERTER = "40000000-0000-0000-0000-000000000001"
BATTERY = "40000000-0000-0000-0000-000000000002"
SOLAR = "40000000-0000-0000-0000-000000000003"
GRID = "40000000-0000-0000-0000-000000000004"
LOAD = "40000000-0000-0000-0000-000000000005"
GENSET = "40000000-0000-0000-0000-000000000006"

ROLES = {
    "inverter": INVERTER,
    "battery": BATTERY,
    "solar_array": SOLAR,
    "grid_connection": GRID,
}


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _synthetic_sg04lp3() -> list[int]:
    """A plausible SG04LP3 register image. SYNTHETIC — never a captured dump.

    Values are chosen so a wrong scale, a wrong word order or a swapped pair
    is obvious in the assertion rather than plausible.
    """
    words = [0] * 1000

    # Identity: device type 0x005 (LV three-phase hybrid), protocol 1.2,
    # serial "2305A" across five ASCII registers.
    words[0] = 0x0005
    words[2] = 0x0102
    words[3:8] = [0x3233, 0x3035, 0x4100, 0x0000, 0x0000]

    words[500] = 2  # running
    words[540] = 1500  # DC transformer 50.0 degC (identity/evidence only)
    words[541] = 1450  # AC radiator 45.0 degC
    words[555:559] = [0, 0, 0, 0]  # no faults

    # Battery: 51.20 V, 88 %, -1500 W, -12.34 A, 31.5 degC
    words[586] = 1315  # 131.5 * 0.1 - 100 = 31.5
    words[587] = 5120  # x0.01 = 51.20 V
    words[588] = 88
    words[590] = 0xFA24  # -1500 W
    words[591] = 0xFB2E  # -1234 -> -12.34 A

    # BMS block
    words[210] = 5760  # 57.60 V charge voltage
    words[212] = 100  # charge current limit
    words[213] = 120  # discharge current limit
    words[220] = 0  # no alarm

    # Grid: 230.5/231.0/229.5 V, 50.02 Hz, +4200 W total (import)
    words[598], words[599], words[600] = 2305, 2310, 2295
    words[609] = 5002
    words[622], words[623], words[624] = 1400, 1400, 1400
    words[625] = 4200

    # Inverter output: 3 x 2500 W at 230 V
    words[627], words[628], words[629] = 2300, 2301, 2302
    words[633], words[634], words[635] = 2500, 2500, 2500

    # Load: 3 x 1200 W, 3600 total
    words[650], words[651], words[652], words[653] = 1200, 1200, 1200, 3600

    # Gen port total 4000 W
    words[667] = 4000

    # PV: MPPT1 3000 W at 380.0 V / 7.9 A, MPPT2 2000 W at 375.0 V / 5.3 A
    words[672], words[673] = 3000, 2000
    words[676], words[677] = 3800, 79
    words[678], words[679] = 3750, 53

    # Energy counters, all uint32 with the LOW word first.
    def u32_le(value: int) -> tuple[int, int]:
        return value & 0xFFFF, (value >> 16) & 0xFFFF

    words[529] = 421  # 42.1 kWh today
    words[534], words[535] = u32_le(123456)  # 12345.6 kWh total PV
    words[516], words[517] = u32_le(54321)  # 5432.1 kWh charged
    words[518], words[519] = u32_le(45678)  # 4567.8 kWh discharged
    words[522], words[523] = u32_le(98765)  # 9876.5 kWh imported
    words[524], words[525] = u32_le(11111)  # 1111.1 kWh exported
    return words


class _Server:
    """A real pymodbus TCP server over a synthetic register image."""

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
            SimDevice(1, simdata=[block]), address=("127.0.0.1", self.port)
        )
        self._task = asyncio.create_task(self._server.serve_forever())
        await asyncio.sleep(0.3)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._server is not None:
            await self._server.shutdown()
        if self._task is not None:
            self._task.cancel()


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


# ── The map polls a real server across four assets ──────────────


@pytest.mark.asyncio
async def test_sg04lp3_polls_a_real_server_with_four_roles_bound() -> None:
    """The spec's WS0 gate: one map, one device, four assets."""
    values, readings = await _poll("deye_sg04lp3", _synthetic_sg04lp3())

    assert values["inverter.ac_output_l1_power_w"] == 2500.0
    assert values["battery.soc_pct"] == 88.0
    assert values["solar.mppt1_power_w"] == 3000.0
    assert values["grid.voltage_v"] == pytest.approx(230.5)

    assets = {r.asset_id for r in readings}
    assert assets == {INVERTER, BATTERY, SOLAR, GRID}


@pytest.mark.asyncio
async def test_every_reading_carries_its_own_assets_id() -> None:
    _, readings = await _poll("deye_sg04lp3", _synthetic_sg04lp3())
    by_metric = {r.metric_name: r for r in readings}

    assert by_metric["battery.voltage_v"].asset_id == BATTERY
    assert by_metric["solar.mppt1_voltage_v"].asset_id == SOLAR
    assert by_metric["grid.frequency_hz"].asset_id == GRID
    assert by_metric["inverter.temperature_c"].asset_id == INVERTER


@pytest.mark.asyncio
async def test_readings_carry_the_register_they_came_from() -> None:
    _, readings = await _poll("deye_sg04lp3", _synthetic_sg04lp3())
    by_metric = {r.metric_name: r for r in readings}
    assert by_metric["battery.soc_pct"].raw_register == "modbus:1:588"


# ── Scaling, offsets and sign ───────────────────────────────────


@pytest.mark.asyncio
async def test_lv_scales_decode_as_the_sources_state_them() -> None:
    values, _ = await _poll("deye_sg04lp3", _synthetic_sg04lp3())

    assert values["battery.voltage_v"] == pytest.approx(51.20)  # x0.01 on LV
    assert values["battery.current_a"] == pytest.approx(-12.34)
    assert values["battery.power_w"] == -1500.0
    assert values["grid.frequency_hz"] == pytest.approx(50.02)
    assert values["solar.mppt1_voltage_v"] == pytest.approx(380.0)
    assert values["solar.mppt1_current_a"] == pytest.approx(7.9)


@pytest.mark.asyncio
async def test_temperature_applies_scale_then_offset() -> None:
    """raw*0.1-100. The other order reads about -1000 degC."""
    values, _ = await _poll("deye_sg04lp3", _synthetic_sg04lp3())
    assert values["inverter.temperature_c"] == pytest.approx(45.0)
    assert values["battery.temperature_c"] == pytest.approx(31.5)


@pytest.mark.asyncio
async def test_the_mppt_sum_is_derived_from_both_strings() -> None:
    values, _ = await _poll("deye_sg04lp3", _synthetic_sg04lp3())
    assert values["solar.mppt1_power_w"] == 3000.0
    assert values["solar.mppt2_power_w"] == 2000.0
    assert values["solar.pv_power_w"] == 5000.0


@pytest.mark.asyncio
async def test_a_missing_string_withholds_the_sum_entirely() -> None:
    """A partial sum still plots as plant power, so there is no partial sum."""
    words = _synthetic_sg04lp3()
    words[673] = 0xFFFF  # MPPT2 present but we disable it below instead
    values, _ = await _poll("deye_sg04lp3", words, registers_disable=["pv2_w"])
    assert values["solar.mppt1_power_w"] == 3000.0
    assert "solar.mppt2_power_w" not in values
    assert "solar.pv_power_w" not in values


# ── Counters: word order is the defect that matters ─────────────


@pytest.mark.asyncio
async def test_lifetime_counters_decode_low_word_first() -> None:
    """Reading these big-endian turns 12345.6 kWh into roughly 806,092 kWh."""
    values, _ = await _poll("deye_sg04lp3", _synthetic_sg04lp3())

    assert values["solar.total_yield_kwh"] == pytest.approx(12345.6)
    assert values["battery.charge_energy_kwh"] == pytest.approx(5432.1)
    assert values["battery.discharge_energy_kwh"] == pytest.approx(4567.8)
    assert values["grid.import_energy_kwh"] == pytest.approx(9876.5)
    assert values["grid.export_energy_kwh"] == pytest.approx(1111.1)


@pytest.mark.asyncio
async def test_the_daily_yield_is_a_separate_resetting_counter() -> None:
    values, _ = await _poll("deye_sg04lp3", _synthetic_sg04lp3())
    assert values["solar.daily_yield_kwh"] == pytest.approx(42.1)


# ── The signed grid CT feeds both directions ────────────────────


@pytest.mark.asyncio
async def test_a_positive_ct_reading_is_import_and_export_is_a_real_zero() -> None:
    values, _ = await _poll("deye_sg04lp3", _synthetic_sg04lp3())
    assert values["grid.import_power_w"] == 4200.0
    assert values["grid.export_power_w"] == 0.0


@pytest.mark.asyncio
async def test_a_negative_ct_reading_becomes_export_as_a_magnitude() -> None:
    words = _synthetic_sg04lp3()
    words[625] = 0xEF98  # -4200
    values, _ = await _poll("deye_sg04lp3", words)
    assert values["grid.import_power_w"] == 0.0
    assert values["grid.export_power_w"] == 4200.0


# ── Optional roles and the gen port ─────────────────────────────


@pytest.mark.asyncio
async def test_the_load_registers_only_appear_when_a_load_asset_is_bound() -> None:
    without, _ = await _poll("deye_sg04lp3", _synthetic_sg04lp3())
    assert "load.power_w" not in without

    with_load, readings = await _poll(
        "deye_sg04lp3",
        _synthetic_sg04lp3(),
        asset_mappings={**ROLES, "load": LOAD},
    )
    assert with_load["load.power_w"] == 3600.0
    assert with_load["load.l1_power_w"] == 1200.0
    assert {r.asset_id for r in readings if r.metric_name.startswith("load.")} == {LOAD}


@pytest.mark.asyncio
async def test_the_gen_port_stays_off_until_it_is_both_bound_and_enabled() -> None:
    """Reading a gen port nothing is wired to publishes a fiction."""
    bound_only, _ = await _poll(
        "deye_sg04lp3",
        _synthetic_sg04lp3(),
        asset_mappings={**ROLES, "diesel_genset": GENSET},
    )
    assert "genset.power_w" not in bound_only

    enabled, _ = await _poll(
        "deye_sg04lp3",
        _synthetic_sg04lp3(),
        asset_mappings={**ROLES, "diesel_genset": GENSET},
        features={"gen_port": True},
    )
    assert enabled["genset.power_w"] == 4000.0


# ── Identity, and the 540/541 decision ──────────────────────────


@pytest.mark.asyncio
async def test_identity_is_read_from_the_device_and_is_not_telemetry() -> None:
    async with _Server(_synthetic_sg04lp3()) as server:
        adapter = GenericModbusAdapter()
        await adapter.initialize(
            {
                "vendor_map": "deye_sg04lp3",
                "protocol": "tcp",
                "host": "127.0.0.1",
                "port": server.port,
                "asset_mappings": dict(ROLES),
            }
        )
        identity = adapter.identity()
        readings = await adapter.poll()
        await adapter.shutdown()

    assert identity["serial"] == "2305A"
    assert identity["device_type"] == "5.0"
    assert not any(r.metric_name.startswith("identity.") for r in readings)


@pytest.mark.asyncio
async def test_register_540_is_evidence_and_only_541_is_the_temperature() -> None:
    """Decision 4: both on one asset would collide on the measurements key."""
    async with _Server(_synthetic_sg04lp3()) as server:
        adapter = GenericModbusAdapter()
        await adapter.initialize(
            {
                "vendor_map": "deye_sg04lp3",
                "protocol": "tcp",
                "host": "127.0.0.1",
                "port": server.port,
                "asset_mappings": dict(ROLES),
            }
        )
        identity = adapter.identity()
        readings = await adapter.poll()
        await adapter.shutdown()

    temps = [r for r in readings if r.metric_name == "inverter.temperature_c"]
    assert len(temps) == 1
    assert temps[0].value == pytest.approx(45.0)  # register 541
    assert identity["dc_transformer_temp_c"] == "50.0"  # register 540


# ── Suspect values are kept, not dropped ────────────────────────


@pytest.mark.asyncio
async def test_a_reading_outside_the_declared_bounds_is_kept_and_flagged() -> None:
    words = _synthetic_sg04lp3()
    words[588] = 150  # SOC of 150 %
    _, readings = await _poll("deye_sg04lp3", words)
    soc = next(r for r in readings if r.metric_name == "battery.soc_pct")

    assert soc.value == 150.0
    assert soc.quality == 1  # suspect


@pytest.mark.asyncio
async def test_an_unmapped_run_state_emits_nothing_rather_than_a_raw_code() -> None:
    words = _synthetic_sg04lp3()
    words[500] = 9  # not in the map's table
    values, _ = await _poll("deye_sg04lp3", words)
    assert "inverter.state" not in values


# ── Fault words ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_fault_bits_set_reports_a_real_zero() -> None:
    values, _ = await _poll("deye_sg04lp3", _synthetic_sg04lp3())
    assert values["inverter.error_code"] == 0.0


@pytest.mark.asyncio
async def test_a_fault_bit_reports_its_position_and_carries_the_raw_words() -> None:
    words = _synthetic_sg04lp3()
    words[556] = 0x0004  # bit 2 of the second word -> position 19
    _, readings = await _poll("deye_sg04lp3", words)
    fault = next(r for r in readings if r.metric_name == "inverter.error_code")

    assert fault.value == 19.0
    assert "0000,0004,0000,0000" in fault.raw_register


# ── HV variant ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_hv_map_overrides_scaling_and_adds_two_more_strings() -> None:
    words = _synthetic_sg04lp3()
    words[674], words[675] = 1500, 1000  # PV3, PV4 exist on HV
    words[224] = 97  # battery SOH, HV only
    values, _ = await _poll("deye_sg01hp3", words)

    # HV: battery voltage x0.1 (not x0.01), power x10, PV x10.
    assert values["battery.voltage_v"] == pytest.approx(512.0)
    assert values["battery.power_w"] == -15000.0
    assert values["solar.mppt1_power_w"] == 30000.0
    assert values["solar.mppt3_power_w"] == 15000.0
    assert values["solar.mppt4_power_w"] == 10000.0
    assert values["solar.pv_power_w"] == 75000.0
    assert values["battery.soh_pct"] == 97.0


@pytest.mark.asyncio
async def test_the_hv_map_inherits_everything_it_does_not_override() -> None:
    values, _ = await _poll("deye_sg01hp3", _synthetic_sg04lp3())
    assert values["grid.frequency_hz"] == pytest.approx(50.02)
    assert values["inverter.ac_output_l1_power_w"] == 2500.0
    assert values["grid.import_energy_kwh"] == pytest.approx(9876.5)


def test_the_hv_map_declares_its_upstream_caveat() -> None:
    loaded = load_vendor_map("deye_sg01hp3")
    assert loaded.extends == ("deye_sg04lp3",)
    assert loaded.verification == "community"
    text = open("collector/adapters/vendor_maps/deye_sg01hp3.yaml").read()
    assert "#558" in text  # the unanswered upstream report about HP3 units


# ── SunSynk alias ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_sunsynk_alias_reads_identically_to_the_deye_map() -> None:
    words = _synthetic_sg04lp3()
    deye, _ = await _poll("deye_sg04lp3", words)
    sunsynk, _ = await _poll("sunsynk_3p", words)
    assert deye == sunsynk


def test_the_alias_changes_only_the_vendor_identity() -> None:
    deye = load_vendor_map("deye_sg04lp3")
    sunsynk = load_vendor_map("sunsynk_3p")

    assert sunsynk.vendor == "sunsynk"
    assert deye.vendor == "deye"
    assert sunsynk.extends == ("deye_sg04lp3",)
    # Same registers, same addresses, same decoding.
    assert {(s.id, s.address, s.data_type, s.scale) for s in sunsynk.all_registers} == {
        (s.id, s.address, s.data_type, s.scale) for s in deye.all_registers
    }


def test_the_alias_inherits_the_community_tier_and_its_caveats() -> None:
    sunsynk = load_vendor_map("sunsynk_3p")
    assert sunsynk.verification == "community"
    assert "open bench questions" in sunsynk.notes


# ── Provenance and safety of the shipped files ──────────────────


@pytest.mark.parametrize("name", ["deye_sg04lp3", "deye_sg01hp3", "sunsynk_3p"])
def test_every_deye_family_map_is_community_and_read_only(name: str) -> None:
    loaded = load_vendor_map(name)
    assert loaded.verification == "community"
    assert loaded.source
    assert all(spec.function_code in (3, 4) for spec in loaded.all_registers)


@pytest.mark.parametrize("name", ["deye_sg04lp3", "deye_sg01hp3", "sunsynk_3p"])
def test_the_unresolved_questions_are_marked_in_the_file(name: str) -> None:
    """A register a source does not settle must say so where it is read."""
    text = open(f"collector/adapters/vendor_maps/{name}.yaml").read()
    if name == "sunsynk_3p":
        # The alias declares no registers of its own; its caveat lives in the
        # notes, which YAML folds across lines, so read the parsed value.
        assert "open bench questions" in load_vendor_map(name).notes
        return
    assert "UNVERIFIED — bench required" in text


@pytest.mark.asyncio
async def test_a_deye_poll_never_writes_to_the_device() -> None:
    """The server would happily accept a write; the adapter must not send one."""
    words = _synthetic_sg04lp3()
    async with _Server(words) as server:
        adapter = GenericModbusAdapter()
        await adapter.initialize(
            {
                "vendor_map": "deye_sg04lp3",
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

        # Re-read through a second adapter: the image is untouched.
        verify = GenericModbusAdapter()
        await verify.initialize(
            {
                "vendor_map": "deye_sg04lp3",
                "protocol": "tcp",
                "host": "127.0.0.1",
                "port": server.port,
                "asset_mappings": dict(ROLES),
            }
        )
        values = {r.metric_name: r.value for r in await verify.poll()}
        await verify.shutdown()

    assert values["battery.soc_pct"] == 88.0


def test_a_site_cannot_add_a_write_path_to_a_deye_map() -> None:
    from collector.adapters.vendor_map import VendorMapError

    with pytest.raises(VendorMapError, match="registers"):
        resolve_vendor_map(
            "deye_sg04lp3",
            {
                "asset_mappings": dict(ROLES),
                "registers": [{"metric": "battery.soc_pct", "address": 1}],
            },
        )
