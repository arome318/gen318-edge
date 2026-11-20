"""ATESS HPS — the vendor-document map, and the conventions it gets right.

The ATESS map is single-sourced from ATESS's own protocol document (V3.22).
Nothing else exists to cross-check it against, so the tests here pin the
places where a wrong guess decodes cleanly:

- the line runs 8N2, not the usual 8N1;
- telemetry is input registers (0x04), not holding registers;
- power is in tenths of a kilowatt, so the scale to watts is 100, not 0.1;
- every 32-bit counter puts the HIGH word at the LOWER address, the
  opposite of the Deye and SmartGen maps;
- signed battery and grid power, and the fault bit words, are left out on
  purpose because the document settles neither the sign nor the byte order.

A test fails if one of those omissions quietly reappears.
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any

import pytest
import yaml

from collector.adapters.generic_modbus import GenericModbusAdapter
from collector.adapters.vendor_map import load_vendor_map

INVERTER = "71000000-0000-0000-0000-000000000001"
BATTERY = "71000000-0000-0000-0000-000000000002"
SOLAR = "71000000-0000-0000-0000-000000000003"
GRID = "71000000-0000-0000-0000-000000000004"
LOAD = "71000000-0000-0000-0000-000000000005"
GENSET = "71000000-0000-0000-0000-000000000006"

ROLES = {
    "inverter": INVERTER,
    "battery": BATTERY,
    "solar_array": SOLAR,
    "grid_connection": GRID,
    "load": LOAD,
}
ROLES_WITH_GENSET = {**ROLES, "diesel_genset": GENSET}

MAP = "atess_hps"


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _be32(value: int) -> tuple[int, int]:
    """High word at the lower address."""
    return (value >> 16) & 0xFFFF, value & 0xFFFF


def _s16(value: int) -> int:
    return value & 0xFFFF


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


async def _poll(
    words: dict[int, int],
    roles: dict[str, str],
    *,
    features: dict[str, bool] | None = None,
) -> dict[str, float]:
    async with _Server(words) as server:
        adapter = GenericModbusAdapter()
        config: dict[str, Any] = {
            "vendor_map": MAP,
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
    return {r.metric_name: r.value for r in readings}


def _words(**overrides: int) -> dict[int, int]:
    pv_hi, pv_lo = _be32(1_234_567)  # 123,456.7 kWh lifetime PV
    dis_hi, dis_lo = _be32(88_000)  # 8,800.0 kWh discharged
    chg_hi, chg_lo = _be32(91_500)  # 9,150.0 kWh charged
    imp_hi, imp_lo = _be32(250_001)  # 25,000.1 kWh from grid
    exp_hi, exp_lo = _be32(70_000)  # 7,000.0 kWh to grid
    dg_hi, dg_lo = _be32(66_000)  # 6,600.0 kWh from DG
    words = {
        0: 6123,  # PV1 612.3 V
        1: 5321,  # battery 532.1 V
        3: 125,  # PV1 12.5 A
        13: 4012,  # grid U-V 401.2 V (line-to-line)
        16: 5000,  # output 50.00 Hz
        21: 4999,  # grid 49.99 Hz
        30: 452,  # converter heat sink 45.2 degC
        47: 81,  # SOC 81 %
        49: 312,  # load 31.2 kW
        51: 76,  # PV1 7.6 kW
        56: 2301,  # output U 230.1 V
        57: 2298,  # output V 229.8 V
        58: 2305,  # output W 230.5 V
        62: 1234,  # PV today 123.4 kWh
        64: pv_hi,
        65: pv_lo,
        68: dis_hi,
        69: dis_lo,
        72: chg_hi,
        73: chg_lo,
        90: imp_hi,
        91: imp_lo,
        96: exp_hi,
        97: exp_lo,
        180: 2,  # running state: on-grid
        220: 185,  # DG 18.5 kW
        224: dg_hi,
        225: dg_lo,
        # Registers the map must NOT read. Non-zero so that a stray read
        # would surface as a reading rather than hide as a zero.
        2: _s16(-120),  # battery current, sign undocumented
        17: _s16(-45),  # battery power, sign undocumented
        19: _s16(33),  # bypass active power, sign undocumented
        158: _s16(71),  # inverter active power, sign undocumented
        181: 0x0102,  # fault word 1, byte order in dispute
    }
    words.update({int(k.lstrip("a")): v for k, v in overrides.items()})
    return words


# ── What the file declares ──────────────────────────────────────


def test_ships_as_community_single_sourced_and_dated() -> None:
    vendor_map = load_vendor_map(MAP)
    assert vendor_map.verification == "community"
    assert vendor_map.complete is True
    assert "V3.22" in vendor_map.source
    assert "2026-09-27" in vendor_map.source


def test_declares_no_write() -> None:
    raw = yaml.safe_load(Path(f"collector/adapters/vendor_maps/{MAP}.yaml").read_text())
    for key in ("write_mappings", "writes", "write_registers", "write"):
        assert key not in raw


def test_line_is_8n2_on_input_registers() -> None:
    """ATESS §2.3.1: 8 data bits, no parity, TWO stop bits; telemetry is 0x04."""
    protocol = load_vendor_map(MAP).protocol
    assert protocol["stopbits"] == 2
    assert protocol["parity"] == "N"
    assert protocol["bytesize"] == 8
    assert protocol["baudrate"] == 9600
    assert protocol["function_code"] == 4


def test_every_register_reads_input_registers() -> None:
    for group in load_vendor_map(MAP).groups:
        for spec in group.registers:
            assert spec.function_code == 4, spec.id


# ── Decoding ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decodes_the_documented_scales() -> None:
    values = await _poll(_words(), ROLES)
    assert values["solar.pv_voltage_v"] == pytest.approx(612.3)
    assert values["solar.pv_current_a"] == pytest.approx(12.5)
    # Tenths of a kilowatt: 76 on the wire is 7,600 W, not 7.6 W.
    assert values["solar.pv_power_w"] == pytest.approx(7600.0)
    assert values["load.power_w"] == pytest.approx(31200.0)
    assert values["battery.voltage_v"] == pytest.approx(532.1)
    assert values["battery.soc_pct"] == pytest.approx(81.0)
    assert values["grid.voltage_v"] == pytest.approx(401.2)
    # Frequency is hundredths.
    assert values["grid.frequency_hz"] == pytest.approx(49.99)
    assert values["inverter.ac_output_freq_hz"] == pytest.approx(50.00)
    assert values["inverter.ac_output_l1_voltage_v"] == pytest.approx(230.1)
    assert values["inverter.ac_output_l2_voltage_v"] == pytest.approx(229.8)
    assert values["inverter.ac_output_l3_voltage_v"] == pytest.approx(230.5)
    assert values["inverter.temperature_c"] == pytest.approx(45.2)
    assert values["inverter.state"] == pytest.approx(2.0)
    assert values["solar.daily_yield_kwh"] == pytest.approx(123.4)


@pytest.mark.asyncio
async def test_counters_put_the_high_word_first() -> None:
    """A value above 65,535 is only right if the word order is right."""
    values = await _poll(_words(), ROLES)
    assert values["solar.total_yield_kwh"] == pytest.approx(123_456.7)
    assert values["battery.discharge_energy_kwh"] == pytest.approx(8_800.0)
    assert values["battery.charge_energy_kwh"] == pytest.approx(9_150.0)
    assert values["grid.import_energy_kwh"] == pytest.approx(25_000.1)
    assert values["grid.export_energy_kwh"] == pytest.approx(7_000.0)


@pytest.mark.asyncio
async def test_negative_temperature_is_signed() -> None:
    values = await _poll(_words(a30=_s16(-50)), ROLES)
    assert values["inverter.temperature_c"] == pytest.approx(-5.0)


@pytest.mark.asyncio
async def test_an_unlisted_running_state_is_refused_not_guessed() -> None:
    values = await _poll(_words(a180=9), ROLES)
    assert "inverter.state" not in values


# ── The deliberate omissions ────────────────────────────────────

ABSENT_METRICS = (
    "battery.current_a",  # register 2: sign undocumented
    "battery.power_w",  # register 17: sign undocumented
    "grid.import_power_w",  # register 19: sign undocumented
    "grid.export_power_w",  # register 19: sign undocumented
    "inverter.ac_output_power_w",  # register 158: sign undocumented
    "inverter.error_code",  # 181-188: "Low byte first" contradicts the rest of the document
)
ABSENT_ADDRESSES = (2, 17, 19, 158, *range(181, 189))


@pytest.mark.parametrize("metric", ABSENT_METRICS)
def test_signed_and_byte_order_disputed_values_are_not_mapped(metric: str) -> None:
    for group in load_vendor_map(MAP).groups:
        for spec in group.registers:
            assert spec.metric != metric, (
                f"{spec.id} maps {metric}; see the file header"
            )


def test_disputed_addresses_are_not_declared() -> None:
    declared = {
        spec.address
        for group in load_vendor_map(MAP).groups
        for spec in group.registers
        if spec.address is not None
    }
    assert declared.isdisjoint(ABSENT_ADDRESSES)


@pytest.mark.asyncio
async def test_a_poll_publishes_none_of_the_omitted_metrics() -> None:
    values = await _poll(_words(), ROLES)
    for metric in ABSENT_METRICS:
        assert metric not in values, metric


# ── The generator input ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_generator_registers_stay_off_by_default() -> None:
    """With nothing on DG the registers read zero; they must not become a genset."""
    values = await _poll(_words(), ROLES)
    assert "genset.power_w" not in values
    assert "genset.energy_kwh" not in values


@pytest.mark.asyncio
async def test_generator_registers_read_when_the_site_declares_one() -> None:
    values = await _poll(_words(), ROLES_WITH_GENSET, features={"generator": True})
    assert values["genset.power_w"] == pytest.approx(18500.0)
    assert values["genset.energy_kwh"] == pytest.approx(6_600.0)
