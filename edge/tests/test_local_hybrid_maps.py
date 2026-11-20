"""GoodWe, Solis and Sungrow local maps (plan Task 11, amendment B).

Amendment B keeps the maps and drops the diesel-savings and parts-forecaster
consumer changes: those touch money and evidence semantics and belong to a
follow-on plan. Nothing here writes a number into a commercial calculation.

The register conventions are what these tests protect. Every one of the four
maps differs from its neighbours in a way that decodes CLEANLY when wrong:
GoodWe answers only at slave 247, Solis's printed addresses are already wire
addresses while Sungrow's are one higher, and GoodWe's 32-bit values are
high-word-first while Sungrow's are low-word-first. A map that gets any of
these backwards produces plausible numbers, which is the worst outcome
available.
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import pytest

from collector.adapters.generic_modbus import GenericModbusAdapter
from collector.adapters.vendor_map import load_vendor_map

INVERTER = "80000000-0000-0000-0000-000000000001"
BATTERY = "80000000-0000-0000-0000-000000000002"
SOLAR = "80000000-0000-0000-0000-000000000003"
GRID = "80000000-0000-0000-0000-000000000004"

FULL_ROLES = {
    "inverter": INVERTER,
    "battery": BATTERY,
    "solar_array": SOLAR,
    "grid_connection": GRID,
}
SG_ROLES = {"inverter": INVERTER, "solar_array": SOLAR}


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _be(value: int) -> tuple[int, int]:
    return (value >> 16) & 0xFFFF, value & 0xFFFF


def _le(value: int) -> tuple[int, int]:
    return value & 0xFFFF, (value >> 16) & 0xFFFF


class _Server:
    """A real pymodbus server serving BOTH register files at one unit id."""

    def __init__(self, words: dict[int, int], unit_id: int = 1) -> None:
        self.words = words
        self.unit_id = unit_id
        self.port = _free_port()
        self._server: Any = None
        self._task: asyncio.Task[Any] | None = None

    async def __aenter__(self) -> _Server:
        from pymodbus.server import ModbusTcpServer
        from pymodbus.simulator import DataType, SimData, SimDevice

        size = max(self.words) + 8 if self.words else 16
        values = [0] * size
        for address, word in self.words.items():
            values[address] = word
        block = SimData(0, values=values, datatype=DataType.REGISTERS)
        self._server = ModbusTcpServer(
            SimDevice(self.unit_id, simdata=[block]), address=("127.0.0.1", self.port)
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
    map_name: str, words: dict[int, int], roles: dict[str, str], unit_id: int = 1
) -> dict[str, float]:
    async with _Server(words, unit_id=unit_id) as server:
        adapter = GenericModbusAdapter()
        assert await adapter.initialize(
            {
                "vendor_map": map_name,
                "protocol": "tcp",
                "host": "127.0.0.1",
                "port": server.port,
                "asset_mappings": roles,
                # The pacing is real in production; tests do not need to wait.
                "request_spacing_ms": 0,
            }
        )
        readings = await adapter.poll()
        await adapter.shutdown()
    return {reading.metric_name: reading.value for reading in readings}


# ── GoodWe ──────────────────────────────────────────────────────


def _goodwe_words() -> dict[int, int]:
    words: dict[int, int] = {
        35103: 3800,  # PV1 380.0 V
        35104: 79,  # PV1 7.9 A
        35107: 3750,  # PV2 375.0 V
        35108: 53,  # PV2 5.3 A
        35121: 2305,  # grid 230.5 V
        35123: 5001,  # 50.01 Hz
        35125: 1400,
        35130: 1400,
        35135: 1400,
        35138: 4200,  # inverter total W
        35140: 800,  # active power, positive
        35174: 425,  # 42.5 degC
        35180: 3200,  # battery 320.0 V
        35181: 0xFFF6,  # raw -10 -> -1.0 A at scale 0.1
        37003: 285,  # 28.5 degC
        37004: 50,
        37005: 60,
        37007: 91,
        37008: 98,
    }
    words[35105], words[35106] = _be(3000)  # PV1 3000 W, high word first
    words[35109], words[35110] = _be(2000)
    words[35182], words[35183] = _be(0x100000000 - 1500)  # -1500 W
    words[35191], words[35192] = _be(123456)  # 12345.6 kWh
    words[35193], words[35194] = _be(421)
    words[35195], words[35196] = _be(11111)
    words[35200], words[35201] = _be(98765)
    return words


@pytest.mark.asyncio
async def test_goodwe_polls_at_slave_247() -> None:
    """Unit 1 gets no reply from these models — it reads as a dead bus."""
    values = await _poll("goodwe_et", _goodwe_words(), FULL_ROLES, unit_id=247)
    assert values["solar.mppt1_voltage_v"] == pytest.approx(380.0)


@pytest.mark.asyncio
async def test_goodwe_at_the_wrong_slave_reads_nothing() -> None:
    values = await _poll("goodwe_et", _goodwe_words(), FULL_ROLES, unit_id=1)
    assert values == {}


@pytest.mark.asyncio
async def test_goodwe_thirty_two_bit_values_are_high_word_first() -> None:
    """The opposite of Deye and Sungrow; swapping turns 12345.6 into 806,092."""
    values = await _poll("goodwe_et", _goodwe_words(), FULL_ROLES, unit_id=247)

    assert values["solar.mppt1_power_w"] == 3000.0
    assert values["solar.total_yield_kwh"] == pytest.approx(12345.6)
    assert values["grid.import_energy_kwh"] == pytest.approx(9876.5)
    assert values["battery.power_w"] == -1500.0


@pytest.mark.asyncio
async def test_goodwe_routes_across_four_assets() -> None:
    values = await _poll("goodwe_et", _goodwe_words(), FULL_ROLES, unit_id=247)

    assert values["inverter.ac_output_power_w"] == 4200.0
    assert values["battery.soc_pct"] == 91.0
    assert values["battery.soh_pct"] == 98.0
    assert values["solar.pv_power_w"] == 5000.0
    assert values["grid.voltage_v"] == pytest.approx(230.5)


@pytest.mark.asyncio
async def test_goodwe_signed_values_keep_their_sign() -> None:
    values = await _poll("goodwe_et", _goodwe_words(), FULL_ROLES, unit_id=247)
    assert values["battery.current_a"] == pytest.approx(-1.0)
    assert values["inverter.temperature_c"] == pytest.approx(42.5)


def test_goodwe_reads_only_the_two_mppts_it_can_trust() -> None:
    """A two-MPPT unit answers the PV3/PV4 addresses with anything at all."""
    loaded = load_vendor_map("goodwe_et")
    ids = {spec.id for spec in loaded.all_registers}
    assert {"pv1_w", "pv2_w"} <= ids
    assert "pv3_w" not in ids


def test_goodwe_block_planning_respects_the_engine_bound() -> None:
    """The vendor's own running-data block is 125 words; the engine caps 120."""
    loaded = load_vendor_map("goodwe_et")
    for group in loaded.groups:
        blocks = GenericModbusAdapter._plan_blocks(
            list(group.registers), max_gap=8, max_words=120
        )
        for _unit, _fc, _start, count, _specs in blocks:
            assert count <= 120, f"{group.name} plans a {count}-word request"


# ── Solis ───────────────────────────────────────────────────────


def _solis_words() -> dict[int, int]:
    words: dict[int, int] = {
        33035: 421,
        33093: 425,
        33094: 5002,
        33128: 2305,
        33129: 1000,
        33133: 512,
        33134: 0xFFF6,  # -1.0 A
        33139: 88,
        33140: 97,
    }
    words[33029], words[33030] = _be(12345)
    words[33057], words[33058] = _be(5000)
    words[33079], words[33080] = _be(4200)
    words[33126], words[33127] = _be(1234567)
    words[33130], words[33131] = _be(800)
    return words


@pytest.mark.asyncio
async def test_solis_printed_addresses_are_the_wire_addresses() -> None:
    """The reference client subtracts nothing — unlike Sungrow, which does."""
    values = await _poll("solis_hybrid_33xxx", _solis_words(), FULL_ROLES)

    assert values["battery.soc_pct"] == 88.0
    assert values["battery.soh_pct"] == 97.0
    assert values["inverter.temperature_c"] == pytest.approx(42.5)


def test_solis_declares_address_base_zero() -> None:
    assert load_vendor_map("solis_hybrid_33xxx").address_base == 0


def test_solis_reads_input_registers_not_holding_registers() -> None:
    """FC04. Reading FC03 at these addresses is a different address space."""
    loaded = load_vendor_map("solis_hybrid_33xxx")
    assert all(spec.function_code == 4 for spec in loaded.all_registers)


@pytest.mark.asyncio
async def test_solis_reports_battery_current_but_not_battery_power() -> None:
    """Direction lives in a separate register, so a magnitude would mislead."""
    values = await _poll("solis_hybrid_33xxx", _solis_words(), FULL_ROLES)

    assert values["battery.current_a"] == pytest.approx(-1.0)
    assert "battery.power_w" not in values


def test_solis_says_why_battery_power_is_absent() -> None:
    text = open("collector/adapters/vendor_maps/solis_hybrid_33xxx.yaml").read()
    assert "33135" in text
    assert "deliberately ABSENT" in text


@pytest.mark.asyncio
async def test_solis_energy_counters_carry_their_own_scales() -> None:
    values = await _poll("solis_hybrid_33xxx", _solis_words(), FULL_ROLES)
    assert values["solar.total_yield_kwh"] == pytest.approx(12345.0)
    assert values["solar.daily_yield_kwh"] == pytest.approx(42.1)
    assert values["meter.energy_kwh"] == pytest.approx(1234.567)


# ── Sungrow SH ──────────────────────────────────────────────────


def _sungrow_sh_words() -> dict[int, int]:
    """Keyed by WIRE address, i.e. the vendor's printed number minus one."""
    words: dict[int, int] = {
        5000: 0x0E00,
        5007: 425,
        5010: 3800,
        5011: 79,
        5012: 3750,
        5013: 53,
        5018: 2305,
        5019: 2310,
        5020: 2295,
        13001: 421,
    }
    words[13009], words[13010] = _le(0x100000000 - 800)  # -800 W (importing)
    words[13002], words[13003] = _le(123456)
    words[13012], words[13013] = _le(54321)
    words[13026], words[13027] = _le(45678)
    return words


@pytest.mark.asyncio
async def test_sungrow_subtracts_one_from_the_printed_address() -> None:
    """The map holds vendor-printed numbers; the engine converts once."""
    values = await _poll("sungrow_sh", _sungrow_sh_words(), FULL_ROLES)

    assert values["solar.mppt1_voltage_v"] == pytest.approx(380.0)
    assert values["inverter.temperature_c"] == pytest.approx(42.5)
    assert values["meter.l1_voltage_v"] == pytest.approx(230.5)


def test_sungrow_declares_address_base_one() -> None:
    """The opposite of Solis, on purpose and from each vendor's own text."""
    assert load_vendor_map("sungrow_sh").address_base == 1
    assert load_vendor_map("sungrow_sg").address_base == 1
    assert load_vendor_map("solis_hybrid_33xxx").address_base == 0


@pytest.mark.asyncio
async def test_sungrow_thirty_two_bit_values_are_low_word_first() -> None:
    values = await _poll("sungrow_sh", _sungrow_sh_words(), FULL_ROLES)

    assert values["solar.total_yield_kwh"] == pytest.approx(12345.6)
    assert values["battery.charge_energy_kwh"] == pytest.approx(5432.1)
    assert values["battery.discharge_energy_kwh"] == pytest.approx(4567.8)


@pytest.mark.asyncio
async def test_sungrows_export_positive_polarity_is_declared_the_other_way() -> None:
    """Its raw register is +export/-import, unlike Deye's and GoodWe's."""
    values = await _poll("sungrow_sh", _sungrow_sh_words(), FULL_ROLES)

    assert values["grid.import_power_w"] == 800.0
    assert values["grid.export_power_w"] == 0.0


@pytest.mark.asyncio
async def test_sungrow_omits_battery_power_because_its_sign_moved() -> None:
    """The source comments out its own sensor: the sign changed with firmware."""
    values = await _poll("sungrow_sh", _sungrow_sh_words(), FULL_ROLES)
    assert "battery.power_w" not in values

    text = open("collector/adapters/vendor_maps/sungrow_sh.yaml").read()
    assert "October 2024 firmware" in text


def test_sungrow_paces_its_requests_for_the_winet_dongle() -> None:
    assert load_vendor_map("sungrow_sh").protocol["request_spacing_ms"] >= 20


# ── Sungrow SG ──────────────────────────────────────────────────


def _sungrow_sg_words() -> dict[int, int]:
    words: dict[int, int] = {
        5000: 0x0E00,
        5002: 421,
        5007: 425,
        5018: 2305,
        5019: 2310,
        5020: 2295,
    }
    words[5003], words[5004] = _le(123456)
    words[5016], words[5017] = _le(9000)
    words[5030], words[5031] = _le(8600)
    return words


@pytest.mark.asyncio
async def test_the_sg_map_reads_a_string_inverter() -> None:
    values = await _poll("sungrow_sg", _sungrow_sg_words(), SG_ROLES)

    assert values["solar.pv_power_w"] == 9000.0
    assert values["inverter.ac_output_power_w"] == 8600.0
    assert values["solar.total_yield_kwh"] == pytest.approx(12345.6)


def test_the_sg_map_has_no_battery_role_at_all() -> None:
    """A string inverter has no battery; a shared map would carry dead rows."""
    loaded = load_vendor_map("sungrow_sg")
    assert "battery" not in loaded.required_roles
    assert "battery" not in loaded.optional_roles
    assert not any(spec.metric.startswith("battery.") for spec in loaded.all_registers)


def test_the_sh_and_sg_maps_are_separate_files() -> None:
    sh = load_vendor_map("sungrow_sh")
    sg = load_vendor_map("sungrow_sg")
    assert sh.extends == ()
    assert sg.extends == ()
    assert sh.model_family != sg.model_family


def test_the_contested_frequency_register_ships_disabled() -> None:
    """Two sources name different registers as canonical; neither is emitted."""
    loaded = load_vendor_map("sungrow_sg")
    assert not any(spec.metric == "grid.frequency_hz" for spec in loaded.all_registers)

    text = open("collector/adapters/vendor_maps/sungrow_sg.yaml").read()
    assert "5148" in text and "5036" in text
    assert "UNVERIFIED — bench required" in text


def test_the_sg_map_omits_the_contested_phase_currents() -> None:
    """ "Documentation says Unsigned, but seems to be returning Signed 16bit"."""
    loaded = load_vendor_map("sungrow_sg")
    assert not any(spec.metric.endswith("_current_a") for spec in loaded.all_registers)


# ── Provenance across all four ──────────────────────────────────

LOCAL_MAPS = ["goodwe_et", "solis_hybrid_33xxx", "sungrow_sh", "sungrow_sg"]


@pytest.mark.parametrize("name", LOCAL_MAPS)
def test_every_local_map_is_community_and_sourced(name: str) -> None:
    loaded = load_vendor_map(name)
    assert loaded.verification == "community"
    assert loaded.source.strip()
    assert all(spec.function_code in (3, 4) for spec in loaded.all_registers)


@pytest.mark.parametrize("name", LOCAL_MAPS)
def test_every_local_map_marks_what_it_could_not_settle(name: str) -> None:
    text = open(f"collector/adapters/vendor_maps/{name}.yaml").read()
    assert "UNVERIFIED — bench required" in text


@pytest.mark.parametrize("name", LOCAL_MAPS)
def test_no_local_map_carries_a_money_or_fuel_metric(name: str) -> None:
    """Amendment B: the consumer changes are a follow-on plan, not this one."""
    loaded = load_vendor_map(name)
    for spec in loaded.all_registers:
        assert not spec.metric.startswith("genset."), spec.metric
        assert "fuel" not in spec.metric, spec.metric
