"""SunSpec meter decode — PCC semantics and per-phase detail (C&I parity WS1e).

A meter mapped with ``asset_mappings: {"pcc": true}`` is the utility boundary:
its energies become grid.import/export (TotWhImp was previously never read —
backwards for an import-dominant C&I site), its signed power becomes the
system grid view, and its voltage/frequency describe the grid itself. A
customer meter (the default) keeps today's names and gains per-phase detail.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from collector.adapters.sunspec_modbus import ModelBlock, SunSpecModbusAdapter


def _meter_registers() -> list[int]:
    """A plausible three-phase model 203 block (offsets per SunSpec)."""
    r = [0] * 107
    r[0:2] = [203, 105]
    # Currents: total + per-phase, scale -1 → 100.0 / 33.3 / 33.4 / 33.3 A
    r[2], r[3], r[4], r[5], r[6] = 1000, 333, 334, 333, 0xFFFF
    # Voltages: avg + per-phase, scale -1 → 230.5 / 230.0 / 231.0 / 230.5 V
    r[7], r[8], r[9], r[10], r[15] = 2305, 2300, 2310, 2305, 0xFFFF
    # Frequency: 50.02 Hz (scale -2)
    r[16], r[17] = 5002, 0xFFFE
    # Power: total -2500 W, phases 100/200/300 W (scale 0)
    r[18], r[19], r[20], r[21], r[22] = 0xF63C, 100, 200, 300, 0
    # Power factor 0.995
    r[33], r[37] = 995, 0xFFFF
    # TotWhExp = 100,000 Wh; TotWhImp = 250,000 Wh; TotWh_SF = 0
    r[38:40] = [0x0001, 0x86A0]
    r[46:48] = [0x0003, 0xD090]
    r[54] = 0
    return r


def _decode(adapter: SunSpecModbusAdapter) -> dict[str, float]:
    readings = adapter._decode_meter(
        _meter_registers(),
        ModelBlock(203, 40138, 105),
        datetime.now(UTC),
    )
    return {reading.metric_name: reading.value for reading in readings}


class TestLegacyCustomerMeter:
    def test_keeps_todays_names_and_gains_phases(self) -> None:
        adapter = SunSpecModbusAdapter()
        adapter._meter_asset_id = "meter-asset"
        values = _decode(adapter)

        # Unchanged legacy surface.
        assert values["meter.current_a"] == 100.0
        assert values["meter.voltage_v"] == 230.5
        assert values["meter.power_w"] == -2500.0
        assert values["meter.power_factor"] == 0.995
        assert values["meter.energy_kwh"] == 100.0
        # New per-phase detail (previously decoded past).
        assert values["meter.l1_voltage_v"] == pytest.approx(230.0)
        assert values["meter.l2_voltage_v"] == pytest.approx(231.0)
        assert values["meter.l3_voltage_v"] == pytest.approx(230.5)
        assert values["meter.l1_current_a"] == pytest.approx(33.3)
        assert values["meter.l2_power_w"] == 200.0
        # A customer meter never claims grid semantics.
        assert not any(name.startswith("grid.") for name in values)
        assert "system.grid_power_w" not in values


class TestPccMeter:
    def test_grid_semantics_replace_meter_names(self) -> None:
        adapter = SunSpecModbusAdapter()
        adapter._meter_asset_id = "pcc-asset"
        adapter._pcc = True
        values = _decode(adapter)

        # Both energy directions, correctly labeled — TotWhImp finally read.
        assert values["grid.export_energy_kwh"] == 100.0
        assert values["grid.import_energy_kwh"] == 250.0
        assert "meter.energy_kwh" not in values
        # Grid electrical state.
        assert values["grid.voltage_v"] == 230.5
        assert values["grid.frequency_hz"] == 50.02
        # Signed PCC power is the system view, not a site meter reading.
        assert values["system.grid_power_w"] == -2500.0
        assert "meter.power_w" not in values
        # Phase detail and PF stay on the meter namespace (the catalog routes
        # meter.* to grid_connection assets too).
        assert values["meter.l2_power_w"] == 200.0
        assert values["meter.power_factor"] == 0.995

    def test_pcc_defaults_off(self) -> None:
        assert SunSpecModbusAdapter()._pcc is False
