"""SRNE, Felicity and SmartGen — the three maps added for Nigerian volume.

Each of these maps leaves something out ON PURPOSE, because the sources
contradict each other on it. Those omissions are the fragile part: they look
exactly like oversights, and the obvious "fix" is to fill them in from
whichever source was read most recently. So the omissions are asserted here,
with the reason attached, and a test fails if one quietly reappears.

The other thing under test is the set of conventions that decode CLEANLY
when wrong. Felicity's battery voltage is scaled by a hundred where almost
every other vendor uses ten. SmartGen runs two stop bits where almost
everything else is 8N1, puts the low word at the lower address where Deep
Sea puts the high word, and reserves 32766 rather than 0xFFFF. Every one of
those produces a plausible number if it is got backwards.
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any

import pytest
import yaml

from collector.adapters.generic_modbus import GenericModbusAdapter
from collector.adapters.vendor_map import load_vendor_map, resolve_vendor_map

INVERTER = "70000000-0000-0000-0000-000000000001"
BATTERY = "70000000-0000-0000-0000-000000000002"
SOLAR = "70000000-0000-0000-0000-000000000003"
GRID = "70000000-0000-0000-0000-000000000004"
LOAD = "70000000-0000-0000-0000-000000000005"
GENSET = "70000000-0000-0000-0000-000000000006"

HYBRID_ROLES = {
    "inverter": INVERTER,
    "battery": BATTERY,
    "solar_array": SOLAR,
    "grid_connection": GRID,
    "load": LOAD,
}
GENSET_ROLES = {"diesel_genset": GENSET, "grid_connection": GRID}

NEW_MAPS = ("srne_hybrid", "felicity_hybrid", "smartgen_hgm")


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _le(value: int) -> tuple[int, int]:
    """Low word at the lower address."""
    return value & 0xFFFF, (value >> 16) & 0xFFFF


class _Server:
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


async def _poll_full(
    map_name: str,
    words: dict[int, int],
    roles: dict[str, str],
    *,
    features: dict[str, bool] | None = None,
    unit_id: int = 1,
) -> list[Any]:
    async with _Server(words, unit_id=unit_id) as server:
        adapter = GenericModbusAdapter()
        config: dict[str, Any] = {
            "vendor_map": map_name,
            "protocol": "tcp",
            "host": "127.0.0.1",
            "port": server.port,
            "asset_mappings": roles,
            "request_spacing_ms": 0,
        }
        if features is not None:
            config["features"] = features
        assert await adapter.initialize(config)
        readings = await adapter.poll()
        await adapter.shutdown()
    return readings


async def _poll(
    map_name: str,
    words: dict[int, int],
    roles: dict[str, str],
    *,
    features: dict[str, bool] | None = None,
    unit_id: int = 1,
) -> dict[str, float]:
    readings = await _poll_full(
        map_name, words, roles, features=features, unit_id=unit_id
    )
    return {r.metric_name: r.value for r in readings}


# ── Every new map, the shared rules ─────────────────────────────


@pytest.mark.parametrize("name", NEW_MAPS)
def test_each_map_ships_as_community_and_never_claims_more(name: str) -> None:
    vendor_map = load_vendor_map(name)
    assert vendor_map.verification == "community"
    assert vendor_map.complete is True


@pytest.mark.parametrize("name", NEW_MAPS)
def test_no_map_declares_a_write(name: str) -> None:
    raw = yaml.safe_load(
        Path(f"collector/adapters/vendor_maps/{name}.yaml").read_text()
    )
    for key in ("write_mappings", "writes", "write_registers", "write"):
        assert key not in raw


@pytest.mark.parametrize("name", NEW_MAPS)
def test_each_map_pins_its_sources_and_dates_them(name: str) -> None:
    vendor_map = load_vendor_map(name)
    assert "2026-09-19" in vendor_map.source
    assert len(vendor_map.source) > 80, "a source line this short cannot be a ledger"


# ── SRNE ────────────────────────────────────────────────────────


def _srne_words() -> dict[int, int]:
    return {
        256: 74,  # SOC 74 %
        257: 532,  # battery 53.2 V
        259: 287,  # battery 28.7 degC
        263: 3801,  # PV1 380.1 V
        264: 72,  # PV1 7.2 A
        265: 2740,  # PV1 2740 W
        270: 2600,  # charge power 2600 W
        271: 3755,  # PV2 375.5 V
        272: 58,  # PV2 5.8 A
        273: 2170,  # PV2 2170 W
        516: 12,  # fault 12 = mains down
        531: 2291,  # grid 229.1 V
        532: 83,  # grid 8.3 A
        533: 4998,  # 49.98 Hz
        534: 2301,  # inverter out 230.1 V
        535: 151,  # 15.1 A
        536: 5001,  # 50.01 Hz
        545: 421,  # heatsink B 42.1 degC
        539: 3410,  # load 3410 W
        61487: 187,  # 18.7 kWh today
    }


@pytest.mark.asyncio
async def test_srne_decodes_its_documented_scales() -> None:
    values = await _poll(
        "srne_hybrid", _srne_words(), HYBRID_ROLES, features={"pv2": True}
    )
    assert values["battery.soc_pct"] == pytest.approx(74.0)
    assert values["battery.voltage_v"] == pytest.approx(53.2)
    assert values["battery.temperature_c"] == pytest.approx(28.7)
    assert values["solar.mppt1_voltage_v"] == pytest.approx(380.1)
    assert values["solar.mppt1_power_w"] == pytest.approx(2740.0)
    # Frequency is hundredths on this vendor, not tenths.
    assert values["grid.frequency_hz"] == pytest.approx(49.98)
    assert values["inverter.ac_output_freq_hz"] == pytest.approx(50.01)
    assert values["inverter.temperature_c"] == pytest.approx(42.1)
    assert values["load.power_w"] == pytest.approx(3410.0)
    assert values["solar.daily_yield_kwh"] == pytest.approx(18.7)


@pytest.mark.asyncio
async def test_srne_omits_battery_current_because_the_sign_is_disputed() -> None:
    """SRNE's own table and the community decoder invert each other."""
    values = await _poll(
        "srne_hybrid", _srne_words(), HYBRID_ROLES, features={"pv2": True}
    )
    assert "battery.current_a" not in values


@pytest.mark.asyncio
async def test_srne_omits_lifetime_energy_because_its_width_is_disputed() -> None:
    """One source reads 0xF038 as 32-bit reversed, another as plain 16-bit."""
    values = await _poll(
        "srne_hybrid", _srne_words(), HYBRID_ROLES, features={"pv2": True}
    )
    assert "solar.total_yield_kwh" not in values
    assert "solar.daily_yield_kwh" in values, "today's counter is 16-bit and agreed"


@pytest.mark.asyncio
async def test_srne_omits_machine_state_because_two_enums_exist() -> None:
    """The official table carries two numberings for the same register."""
    values = await _poll(
        "srne_hybrid", _srne_words(), HYBRID_ROLES, features={"pv2": True}
    )
    assert "inverter.state" not in values


def test_srne_says_why_each_omission_is_an_omission() -> None:
    text = Path("collector/adapters/vendor_maps/srne_hybrid.yaml").read_text()
    for marker in ("0x0102", "0xF038", "0x010B", "0x0210"):
        assert marker in text, f"{marker} is dropped but the map never says why"


@pytest.mark.asyncio
async def test_srne_single_mppt_units_never_see_the_pv2_registers() -> None:
    """Those models answer a read spanning PV2 with exception 0x02."""
    values = await _poll(
        "srne_hybrid", _srne_words(), HYBRID_ROLES, features={"pv2": False}
    )
    assert "solar.mppt2_voltage_v" not in values
    assert "solar.pv_power_w" not in values, "the array total needs both trackers"
    assert values["solar.mppt1_power_w"] == pytest.approx(2740.0)


@pytest.mark.asyncio
async def test_srne_derives_the_array_total_rather_than_reading_it() -> None:
    values = await _poll(
        "srne_hybrid", _srne_words(), HYBRID_ROLES, features={"pv2": True}
    )
    assert values["solar.pv_power_w"] == pytest.approx(2740.0 + 2170.0)


def test_srne_never_asks_a_single_mppt_unit_for_the_pv2_registers() -> None:
    """The feature flag has to reach BLOCK PLANNING, not just the decode.

    A single-MPPT SRNE answers any read that SPANS the PV2 registers with
    exception 0x02 — which takes the whole block down, not just PV2. So it is
    not enough to drop those registers from the results; no request may
    overlap 0x010F-0x0111 (271-273) at all. That property is emergent from
    the flag and the block planner together, so raising max_gap or adding a
    register just past 0x010E would silently undo it.
    """
    adapter = GenericModbusAdapter()
    protocol = load_vendor_map("srne_hybrid").protocol

    def spans(pv2: bool) -> list[tuple[int, int]]:
        resolved = resolve_vendor_map(
            "srne_hybrid",
            {
                "vendor_map": "srne_hybrid",
                "asset_mappings": dict(HYBRID_ROLES),
                "features": {"pv2": pv2},
            },
        )
        out = []
        for group in resolved.groups:
            specs = [r for r in group.registers if r.derived is None]
            if not specs:
                continue
            for _unit, _fc, start, count, _regs in adapter._plan_blocks(
                specs,
                max_gap=protocol["max_gap"],
                max_words=protocol["max_block_words"],
            ):
                out.append((start, start + count - 1))
        return out

    pv2_registers = range(271, 274)
    off = spans(pv2=False)
    assert off, "expected the map to plan at least one request"
    for start, last in off:
        assert not any(start <= address <= last for address in pv2_registers), (
            f"request {start}-{last} spans the PV2 registers on a unit that has none"
        )

    # And with the flag on, they ARE read — otherwise the test above would
    # pass just as well against a map that dropped PV2 entirely.
    on = spans(pv2=True)
    assert any(
        start <= address <= last for start, last in on for address in pv2_registers
    )


def test_srne_honours_the_protocols_own_request_cap() -> None:
    protocol = load_vendor_map("srne_hybrid").protocol
    assert protocol["max_block_words"] == 32
    assert protocol["baudrate"] == 9600
    assert protocol["unit_id"] == 1
    assert protocol["address_base"] == 0


def test_srne_isolates_the_fault_block_that_answers_with_silence() -> None:
    """Some firmware times out on this range instead of erroring."""
    groups = {g.name: g for g in load_vendor_map("srne_hybrid").groups}
    assert groups["faults"].poll_interval_seconds == 300
    assert groups["power"].poll_interval_seconds == 30
    assert [r.metric for r in groups["faults"].registers] == ["inverter.error_code"]


# ── Felicity ────────────────────────────────────────────────────


def _felicity_words() -> dict[int, int]:
    return {
        4355: 0,  # no fault
        4360: 5318,  # battery 53.18 V  (note: scaled by 100, not 10)
        4362: 0x10000 - 1800,  # -1800 W, discharging
        4366: 5002,  # 50.02 Hz
        4369: 2303,  # AC out 230.3 V
        4375: 2288,  # grid 228.8 V
        4377: 4997,  # grid 49.97 Hz
        4382: 1850,  # load 1850 W
        4386: 42,  # inverter heatsink 42 degC (no scale on this register)
        4387: 29,  # battery 29 degC
        4390: 3702,  # PV1 370.2 V
        4393: 81,  # PV1 8.1 A
        4394: 2990,  # PV1 2990 W
        4502: 74,  # AC in 7.4 A
        4503: 80,  # AC out 8.0 A
        4509: 1700,  # AC in 1700 W
        4624: 762,  # SOC 76.2 %
        4625: 981,  # SOH 98.1 %
    }


@pytest.mark.asyncio
async def test_felicity_decodes_the_registers_both_sources_agree_on() -> None:
    values = await _poll("felicity_hybrid", _felicity_words(), HYBRID_ROLES)
    assert values["inverter.ac_output_voltage_v"] == pytest.approx(230.3)
    assert values["grid.voltage_v"] == pytest.approx(228.8)
    assert values["load.power_w"] == pytest.approx(1850.0)
    assert values["solar.mppt1_voltage_v"] == pytest.approx(370.2)
    assert values["solar.mppt1_power_w"] == pytest.approx(2990.0)
    # Neither temperature register carries a scale in the source table.
    assert values["inverter.temperature_c"] == pytest.approx(42.0)
    assert values["battery.temperature_c"] == pytest.approx(29.0)


@pytest.mark.asyncio
async def test_felicity_battery_voltage_is_scaled_by_a_hundred_not_ten() -> None:
    """The scale that would read 531.8 V on a 48 V bank if got wrong."""
    values = await _poll("felicity_hybrid", _felicity_words(), HYBRID_ROLES)
    assert values["battery.voltage_v"] == pytest.approx(53.18)


@pytest.mark.asyncio
async def test_felicity_battery_power_is_signed_with_charge_positive() -> None:
    values = await _poll("felicity_hybrid", _felicity_words(), HYBRID_ROLES)
    assert values["battery.power_w"] == pytest.approx(-1800.0)


@pytest.mark.asyncio
async def test_felicity_omits_battery_current_because_three_registers_claim_it() -> (
    None
):
    values = await _poll("felicity_hybrid", _felicity_words(), HYBRID_ROLES)
    assert "battery.current_a" not in values
    assert "battery.power_w" in values, "power is what consumers read"


@pytest.mark.asyncio
async def test_felicity_publishes_no_energy_counter_at_all() -> None:
    """Neither source found one; the reference app integrates power instead."""
    values = await _poll("felicity_hybrid", _felicity_words(), HYBRID_ROLES)
    assert "solar.daily_yield_kwh" not in values
    assert "solar.total_yield_kwh" not in values


def test_felicity_has_no_identity_block_because_the_wire_carries_none() -> None:
    """Felicity exposes model and serial through its cloud API, not Modbus."""
    assert load_vendor_map("felicity_hybrid").identity == ()


@pytest.mark.asyncio
async def test_felicity_treats_the_documented_sentinel_as_absent() -> None:
    words = _felicity_words()
    words[4624] = 32767  # the pack variant does not populate SOC
    values = await _poll("felicity_hybrid", words, HYBRID_ROLES)
    assert "battery.soc_pct" not in values
    assert values["battery.soh_pct"] == pytest.approx(98.1)


def test_felicity_defaults_to_the_baud_rate_of_the_named_model() -> None:
    """Two sources disagree, 9600 vs 2400; the named model wins the default."""
    assert load_vendor_map("felicity_hybrid").protocol["baudrate"] == 9600


def test_a_site_can_override_the_disputed_baud_rate() -> None:
    resolved = resolve_vendor_map(
        "felicity_hybrid",
        {
            "vendor_map": "felicity_hybrid",
            "baudrate": 2400,
            "asset_mappings": dict(HYBRID_ROLES),
        },
    )
    assert resolved.protocol["baudrate"] == 2400


def test_felicity_records_that_the_port_is_rs232_not_rs485() -> None:
    """An RS485 converter on this socket reads nothing at all."""
    text = Path("collector/adapters/vendor_maps/felicity_hybrid.yaml").read_text()
    assert "RS232, NOT RS485" in text


# ── SmartGen ────────────────────────────────────────────────────


def _smartgen_words() -> dict[int, int]:
    words: dict[int, int] = {
        55: 415,  # mains 415 V line-to-line
        64: 4996,  # mains 49.96 Hz
        75: 413,  # gen 413 V
        84: 5004,  # gen 50.04 Hz
        142: 271,  # starting battery 27.1 V
        149: 84,  # 84 degC
        151: 412,  # 412 kPa
        153: 68,  # 68 % fuel
        170: 452,  # 45.2 L/h
        189: 9,  # normal running
        199: 4312,  # 4312 run hours
        202: 891,  # 891 starts
    }
    words[109], words[110] = _le(2000)  # 2000 tenths of kW = 200 kW
    words[171], words[172] = _le(184300)  # 184300 L lifetime
    words[203], words[204] = _le(1234567)  # 123456.7 kWh
    return words


@pytest.mark.asyncio
async def test_smartgen_decodes_the_two_agreeing_documents() -> None:
    values = await _poll("smartgen_hgm", _smartgen_words(), GENSET_ROLES)
    assert values["genset.output_voltage_v"] == pytest.approx(413.0)
    assert values["genset.output_frequency_hz"] == pytest.approx(50.04)
    assert values["genset.starting_battery_voltage_v"] == pytest.approx(27.1)
    assert values["genset.oil_pressure_kpa"] == pytest.approx(412.0)
    assert values["genset.fuel_level_pct"] == pytest.approx(68.0)
    assert values["genset.coolant_temperature_c"] == pytest.approx(84.0)
    assert values["genset.status"] == pytest.approx(9.0)
    assert values["genset.runtime_hours"] == pytest.approx(4312.0)
    assert values["genset.starts"] == pytest.approx(891.0)
    assert values["grid.voltage_v"] == pytest.approx(415.0)


@pytest.mark.asyncio
async def test_smartgen_power_is_low_word_first_and_converted_to_watts() -> None:
    """Tenths of a kilowatt on the wire; the platform's unit is watts."""
    values = await _poll("smartgen_hgm", _smartgen_words(), GENSET_ROLES)
    assert values["genset.power_w"] == pytest.approx(200_000.0)
    assert values["genset.energy_kwh"] == pytest.approx(123_456.7)
    assert values["genset.fuel_consumed_l"] == pytest.approx(184_300.0)


@pytest.mark.asyncio
async def test_a_reversed_word_order_is_flagged_suspect_not_trusted() -> None:
    """The bound is the guard for the one thing no document proves per-family.

    A controller that actually sent the high word first would decode to
    roughly 13 GW here. That still reaches the series, but it arrives with
    quality=suspect, so it does not count as trusted telemetry.
    """
    words = _smartgen_words()
    words[109], words[110] = words[110], words[109]  # as if high word first
    readings = await _poll_full("smartgen_hgm", words, GENSET_ROLES)
    power = next(r for r in readings if r.metric_name == "genset.power_w")
    assert power.value > 1e9
    assert power.quality == 1, (
        "a physically impossible genset output must not read good"
    )

    good = await _poll_full("smartgen_hgm", _smartgen_words(), GENSET_ROLES)
    real = next(r for r in good if r.metric_name == "genset.power_w")
    assert real.quality == 0


@pytest.mark.asyncio
async def test_smartgen_treats_32766_as_no_normal_data() -> None:
    """Its own documents say so; 0xFFFF means nothing to this vendor."""
    words = _smartgen_words()
    words[151] = 32766  # no oil pressure sender fitted
    values = await _poll("smartgen_hgm", words, GENSET_ROLES)
    assert "genset.oil_pressure_kpa" not in values
    assert values["genset.fuel_level_pct"] == pytest.approx(68.0)


@pytest.mark.asyncio
async def test_smartgen_run_hours_is_one_register_not_a_pair() -> None:
    """The compact SmartGen families split it; this one does not."""
    runtime = next(
        r
        for g in load_vendor_map("smartgen_hgm").groups
        for r in g.registers
        if r.metric == "genset.runtime_hours"
    )
    assert runtime.word_count == 1
    assert runtime.address == 199


def test_smartgen_runs_two_stop_bits_where_almost_everything_else_runs_one() -> None:
    protocol = load_vendor_map("smartgen_hgm").protocol
    assert protocol["stopbits"] == 2
    assert protocol["baudrate"] == 9600
    assert protocol["parity"] == "N"
    assert protocol["address_base"] == 0, "the PLC 4xxxx column is not the wire address"


def test_smartgen_publishes_no_load_percentage() -> None:
    """Neither document has the register, and deriving one needs a rating."""
    metrics = {
        r.metric for g in load_vendor_map("smartgen_hgm").groups for r in g.registers
    }
    assert "genset.load_pct" not in metrics
    assert "genset.power_w" in metrics


def test_smartgen_omits_phase_current_which_removes_the_ratio_dispute() -> None:
    """The one unresolved scale factor sits on registers nothing reads."""
    text = Path("collector/adapters/vendor_maps/smartgen_hgm.yaml").read_text()
    assert "0095-0098" in text
    metrics = {
        r.metric for g in load_vendor_map("smartgen_hgm").groups for r in g.registers
    }
    assert not any("current" in m for m in metrics)


def test_smartgen_names_the_controllers_that_cannot_speak_modbus_at_all() -> None:
    """HGM6110N/6120N have no RS485 port. That is a hardware fact.

    It has to be stated where somebody reading the map will hit it, because
    the alternative is an installer being told the integration is "supported"
    for a controller that physically cannot answer.
    """
    text = Path("collector/adapters/vendor_maps/smartgen_hgm.yaml").read_text()
    assert "HGM6110N and HGM6120N HAVE NO RS485 PORT" in text
    assert "HGM6110NC/6120NC has RS485 port, HGM6110N/6120N without." in text
    assert "HGM7110N and HGM7120N are also NOT mapped" in text


@pytest.mark.asyncio
async def test_a_prime_power_set_with_no_mains_drops_the_mains_readings() -> None:
    values = await _poll("smartgen_hgm", _smartgen_words(), {"diesel_genset": GENSET})
    assert "grid.voltage_v" not in values
    assert values["genset.power_w"] == pytest.approx(200_000.0)


@pytest.mark.asyncio
async def test_smartgen_routes_generator_and_mains_to_different_assets() -> None:
    readings = await _poll_full("smartgen_hgm", _smartgen_words(), GENSET_ROLES)
    by_metric = {r.metric_name: r.asset_id for r in readings}
    assert by_metric["genset.power_w"] == GENSET
    assert by_metric["grid.voltage_v"] == GRID
