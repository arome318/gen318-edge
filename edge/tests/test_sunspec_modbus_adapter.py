"""SunSpec model-discovery and scaling conformance fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from collector.adapters.sunspec_modbus import ModelBlock, SunSpecModbusAdapter


@pytest.mark.asyncio
async def test_model_chain_skips_unknown_models_by_declared_length() -> None:
    adapter = SunSpecModbusAdapter()
    adapter._read_words = AsyncMock(
        side_effect=[
            [1, 66],
            [777, 4],
            [103, 50],
            [0xFFFF, 0],
        ]
    )

    models = await adapter._discover_models(40002)

    assert [model.model_id for model in models] == [1, 777, 103]
    assert [model.address for model in models] == [40002, 40070, 40076]


@pytest.mark.asyncio
async def test_model_chain_rejects_unbounded_vendor_length() -> None:
    adapter = SunSpecModbusAdapter()
    adapter._read_words = AsyncMock(return_value=[777, 513])

    with pytest.raises(ValueError, match="invalid SunSpec model length"):
        await adapter._discover_models(40002)


def test_three_phase_inverter_decodes_signed_values_and_scale_factors() -> None:
    adapter = SunSpecModbusAdapter()
    adapter._inverter_asset_id = "inverter-asset"
    registers = [0] * 52
    registers[0:2] = [103, 50]
    registers[2], registers[6] = 1234, 0x8000  # unavailable scale => omitted current
    registers[10], registers[13] = 2301, 0x8000  # unavailable point
    registers[14], registers[15] = 0xFB2E, 0  # -1234 W
    registers[16], registers[17] = 5000, 0xFFFE  # 50.00 Hz
    registers[33], registers[37] = 425, 0x8000  # unavailable temp
    registers[38] = 4
    registers[40:42] = [0, 0x0004]

    readings = adapter._decode_inverter(
        registers,
        ModelBlock(103, 40070, 50),
        datetime.now(UTC),
    )
    values = {reading.metric_name: reading.value for reading in readings}

    assert values["inverter.ac_output_power_w"] == -1234.0
    assert values["inverter.ac_output_freq_hz"] == 50.0
    assert values["inverter.state"] == 4.0
    assert values["inverter.error_code"] == 4.0
    assert "inverter.ac_output_current_a" not in values
    assert all(reading.asset_id == "inverter-asset" for reading in readings)
    assert all(reading.raw_register.startswith("SunSpec:103:") for reading in readings)


def test_meter_decodes_power_factor_and_accumulated_energy() -> None:
    adapter = SunSpecModbusAdapter()
    adapter._meter_asset_id = "meter-asset"
    registers = [0] * 107
    registers[0:2] = [203, 105]
    registers[2], registers[6] = 0x8000, 0  # unavailable signed current
    registers[7], registers[15] = 2305, 0xFFFF  # 230.5 V (scale -1)
    registers[18], registers[22] = 0xF63C, 0  # -2500 W
    registers[33], registers[37] = 995, 0xFFFF  # 0.995 after % conversion
    registers[38:40] = [0x0001, 0x86A0]  # 100,000 Wh
    registers[54] = 0

    readings = adapter._decode_meter(
        registers,
        ModelBlock(203, 40138, 105),
        datetime.now(UTC),
    )
    values = {reading.metric_name: reading.value for reading in readings}

    assert values == {
        "meter.voltage_v": 230.5,
        "meter.power_w": -2500.0,
        "meter.power_factor": 0.995,
        "meter.energy_kwh": 100.0,
        # Per-phase detail decodes too (C&I parity); this fixture's phase
        # registers legitimately read zero with valid scale factors, and a
        # real 0 A / 0 V / 0 W is data, not absence.
        "meter.l1_current_a": 0.0,
        "meter.l2_current_a": 0.0,
        "meter.l3_current_a": 0.0,
        "meter.l1_voltage_v": 0.0,
        "meter.l2_voltage_v": 0.0,
        "meter.l3_voltage_v": 0.0,
        "meter.l1_power_w": 0.0,
        "meter.l2_power_w": 0.0,
        "meter.l3_power_w": 0.0,
    }
    assert all(reading.asset_id == "meter-asset" for reading in readings)


def test_inverter_event_bitfield_only_omits_exact_acc32_na_sentinel() -> None:
    adapter = SunSpecModbusAdapter()
    adapter._inverter_asset_id = "inverter-asset"
    registers = [0xFFFF] * 52
    registers[0:2] = [103, 50]
    registers[38] = 0xFFFF  # enum16 not implemented
    registers[40:42] = [0xFFFF, 0x0001]  # valid high event bits, not 0xFFFFFFFF

    readings = adapter._decode_inverter(
        registers,
        ModelBlock(103, 40070, 50),
        datetime.now(UTC),
    )
    values = {reading.metric_name: reading.value for reading in readings}

    assert "inverter.state" not in values
    assert values["inverter.error_code"] == float(0xFFFF0001)

    registers[40:42] = [0xFFFF, 0xFFFF]
    readings = adapter._decode_inverter(
        registers,
        ModelBlock(103, 40070, 50),
        datetime.now(UTC),
    )
    assert "inverter.error_code" not in {reading.metric_name for reading in readings}


def test_asset_mapping_rejects_a_non_uuid_identifier() -> None:
    assert SunSpecModbusAdapter._valid_asset_id("not-an-asset") is False
    assert (
        SunSpecModbusAdapter._valid_asset_id("10000000-0000-0000-0000-000000000001")
        is True
    )
