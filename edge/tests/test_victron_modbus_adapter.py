"""Tests for VictronModbusAdapter (Doc 15 §7.3, F-HW-03)."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collector.adapters.base import AdapterStatus
from collector.adapters.victron_modbus import MODBUS_REGISTERS, VictronModbusAdapter

MODBUS_CONFIG: dict = {
    "host": "192.168.1.1",
    "port": 502,
    "poll_interval_seconds": 60,
    "timeout_ms": 3000,
    "retries": 3,
    "unit_ids": [100, 225, 246, 247],
    "asset_mappings": {
        "100": "asset-uuid-system",
        "225": "asset-uuid-battery",
        "246": "asset-uuid-inverter",
        "247": "asset-uuid-solar",
    },
    "write_mappings": {
        "battery.soc_pct": {
            "address": 843,
            "unit_id": 100,
            "type": "uint16",
            "scale": 1.0,
        },
        "inverter.ac_input_power_w": {
            "address": 12,
            "unit_id": 246,
            "type": "int16",
            "scale": 0.1,
        },
    },
}


def _make_register_result(registers: list[int]) -> MagicMock:
    result = MagicMock()
    result.isError.return_value = False
    result.registers = registers
    return result


def _make_error_result() -> MagicMock:
    result = MagicMock()
    result.isError.return_value = True
    return result


def _create_adapter_with_mock() -> tuple[VictronModbusAdapter, AsyncMock]:
    """Create an adapter with a mocked pymodbus client."""
    adapter = VictronModbusAdapter()
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock(return_value=True)
    mock_client.connected = True
    mock_client.close = MagicMock()
    return adapter, mock_client


# ── Initialize Tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_initialize_success() -> None:
    adapter = VictronModbusAdapter()

    with patch("collector.adapters.victron_modbus.AsyncModbusTcpClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.connected = True
        MockClient.return_value = mock_client

        result = await adapter.initialize(MODBUS_CONFIG)

    assert result is True
    assert adapter._status == AdapterStatus.CONNECTED

    call_kwargs = MockClient.call_args.kwargs
    assert call_kwargs["host"] == "192.168.1.1"
    assert call_kwargs["reconnect_delay"] == 1
    assert call_kwargs["reconnect_delay_max"] == 60


@pytest.mark.asyncio
async def test_initialize_failure() -> None:
    adapter = VictronModbusAdapter()

    with patch("collector.adapters.victron_modbus.AsyncModbusTcpClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        result = await adapter.initialize(MODBUS_CONFIG)

    assert result is False
    assert adapter._status == AdapterStatus.ERROR


# ── Poll Tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_poll_reads_all_unit_ids() -> None:
    """Poll reads registers from all 4 configured unit IDs."""
    adapter, mock_client = _create_adapter_with_mock()

    # Every register read returns a valid result
    mock_client.read_holding_registers = AsyncMock(
        return_value=_make_register_result([1000])
    )

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(MODBUS_CONFIG)
        readings = await adapter.poll()

    # Total registers across all 4 unit IDs
    expected_count = sum(
        len(regs)
        for uid, regs in MODBUS_REGISTERS.items()
        if uid in [100, 225, 246, 247]
    )
    assert len(readings) == expected_count

    # Check asset_id mapping
    system_readings = [r for r in readings if r.asset_id == "asset-uuid-system"]
    assert len(system_readings) == len(MODBUS_REGISTERS[100])

    battery_readings = [r for r in readings if r.asset_id == "asset-uuid-battery"]
    assert len(battery_readings) == len(MODBUS_REGISTERS[225])


@pytest.mark.asyncio
async def test_poll_int16_signed_conversion() -> None:
    """Register value 65535 for int16 field reads as -1."""
    adapter, mock_client = _create_adapter_with_mock()

    # Only poll unit 100 for simplicity
    config = {**MODBUS_CONFIG, "unit_ids": [100]}
    mock_client.read_holding_registers = AsyncMock(
        return_value=_make_register_result([65535])
    )

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(config)
        readings = await adapter.poll()

    # battery.current_a (int16) should be -1 / 10 = -0.1
    current_readings = [r for r in readings if r.metric_name == "battery.current_a"]
    assert len(current_readings) >= 1

    # With 65535 as raw, int16 conversion gives -1, then /10 = -0.1
    current = current_readings[0]
    assert current.value == pytest.approx(-0.1, abs=0.01)


@pytest.mark.asyncio
async def test_poll_scale_divide_by_10() -> None:
    """Scale factor 10 divides raw value by 10."""
    adapter, mock_client = _create_adapter_with_mock()

    config = {**MODBUS_CONFIG, "unit_ids": [100]}
    # battery.voltage_v: scale=10, type=uint16
    mock_client.read_holding_registers = AsyncMock(
        return_value=_make_register_result([480])  # 480/10 = 48.0V
    )

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(config)
        readings = await adapter.poll()

    voltage = [r for r in readings if r.metric_name == "battery.voltage_v"]
    assert len(voltage) >= 1
    assert voltage[0].value == pytest.approx(48.0)


@pytest.mark.asyncio
async def test_poll_scale_multiply() -> None:
    """Scale factor 0.01 multiplies raw value by 0.01."""
    adapter, mock_client = _create_adapter_with_mock()

    config = {**MODBUS_CONFIG, "unit_ids": [100]}
    # battery.time_to_go_s: scale=0.01, type=uint16
    mock_client.read_holding_registers = AsyncMock(
        return_value=_make_register_result([36000])  # 36000*0.01 = 360.0
    )

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(config)
        readings = await adapter.poll()

    ttg = [r for r in readings if r.metric_name == "battery.time_to_go_s"]
    assert len(ttg) >= 1
    assert ttg[0].value == pytest.approx(360.0)


@pytest.mark.asyncio
async def test_poll_scale_negative() -> None:
    """Negative scale divides by absolute value."""
    adapter, mock_client = _create_adapter_with_mock()

    config = {**MODBUS_CONFIG, "unit_ids": [100]}
    # battery.consumed_ah: scale=-10, type=uint16
    mock_client.read_holding_registers = AsyncMock(
        return_value=_make_register_result([500])  # 500/10 = 50.0
    )

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(config)
        readings = await adapter.poll()

    consumed = [r for r in readings if r.metric_name == "battery.consumed_ah"]
    assert len(consumed) >= 1
    assert consumed[0].value == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_poll_bounds_validation_suspect() -> None:
    """SoC > 100 returns quality=1 (suspect)."""
    adapter, mock_client = _create_adapter_with_mock()

    config = {**MODBUS_CONFIG, "unit_ids": [100]}
    # battery.soc_pct: scale=1, type=uint16
    mock_client.read_holding_registers = AsyncMock(
        return_value=_make_register_result([150])  # 150 > 100
    )

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(config)
        readings = await adapter.poll()

    soc = [r for r in readings if r.metric_name == "battery.soc_pct"]
    assert len(soc) >= 1
    assert soc[0].quality == 1  # suspect


@pytest.mark.asyncio
async def test_poll_bounds_validation_good() -> None:
    """SoC = 75 returns quality=0 (good)."""
    adapter, mock_client = _create_adapter_with_mock()

    config = {**MODBUS_CONFIG, "unit_ids": [100]}
    mock_client.read_holding_registers = AsyncMock(
        return_value=_make_register_result([75])
    )

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(config)
        readings = await adapter.poll()

    soc = [r for r in readings if r.metric_name == "battery.soc_pct"]
    assert len(soc) >= 1
    assert soc[0].quality == 0  # good


@pytest.mark.asyncio
async def test_poll_register_error_skipped() -> None:
    """Registers returning errors are skipped (error count incremented)."""
    adapter, mock_client = _create_adapter_with_mock()

    config = {**MODBUS_CONFIG, "unit_ids": [100]}
    call_count = 0

    async def alternating_results(**kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 0:
            return _make_error_result()
        return _make_register_result([100])

    mock_client.read_holding_registers = AsyncMock(side_effect=alternating_results)

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(config)
        readings = await adapter.poll()

    # Half should succeed
    total_regs = len(MODBUS_REGISTERS[100])
    assert len(readings) > 0
    assert len(readings) < total_regs


@pytest.mark.asyncio
async def test_poll_disconnected_returns_empty() -> None:
    """Returns empty when client is disconnected."""
    adapter, mock_client = _create_adapter_with_mock()
    mock_client.connected = False

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(MODBUS_CONFIG)
        mock_client.connected = False
        readings = await adapter.poll()

    assert readings == []


# ── Battery Lifetime Counter Tests (unit 225 history registers) ──


@pytest.mark.asyncio
async def test_poll_battery_cycles_decode() -> None:
    """Unit 225 reg 284 (/History/ChargeCycles) decodes uint16 at scale 1."""
    adapter, mock_client = _create_adapter_with_mock()

    config = {**MODBUS_CONFIG, "unit_ids": [225]}
    mock_client.read_holding_registers = AsyncMock(
        return_value=_make_register_result([1234])
    )

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(config)
        readings = await adapter.poll()

    cycles = [r for r in readings if r.metric_name == "battery.cycles"]
    assert len(cycles) == 1
    assert cycles[0].value == pytest.approx(1234.0)
    assert cycles[0].quality == 0  # within (0, 10M) bounds
    assert cycles[0].raw_register == "uid:225/reg:284"


def test_battery_cycles_bounds() -> None:
    """battery.cycles bounds mirror the cloud catalog (0..10M, COUNTER)."""
    assert VictronModbusAdapter._validate_reading("battery.cycles", 65535.0) == 0
    assert VictronModbusAdapter._validate_reading("battery.cycles", 0.0) == 0
    assert VictronModbusAdapter._validate_reading("battery.cycles", 10_000_001.0) == 1
    assert VictronModbusAdapter._validate_reading("battery.cycles", -1.0) == 1


@pytest.mark.asyncio
async def test_read_register_uint32_big_endian() -> None:
    """uint32 decodes big-endian word order: lower address = most significant
    word (Victron dbus_modbustcp mappings.cpp: offset==0 → v>>16)."""
    adapter, mock_client = _create_adapter_with_mock()

    # [hi, lo] — register N carries the MSW, register N+1 the LSW
    mock_client.read_holding_registers = AsyncMock(
        return_value=_make_register_result([0x0001, 0x0002])
    )

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(MODBUS_CONFIG)
        value = await adapter._read_register(
            unit_id=225, address=330, data_type="uint32", scale=10
        )

    assert value == pytest.approx(((0x0001 << 16) | 0x0002) / 10)  # 6553.8
    call_kwargs = mock_client.read_holding_registers.call_args.kwargs
    assert call_kwargs["count"] == 2
    assert call_kwargs["address"] == 330


# ── Write Tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_success() -> None:
    """Write publishes to correct register with inverse scaling."""
    adapter, mock_client = _create_adapter_with_mock()
    write_result = MagicMock()
    write_result.isError.return_value = False
    mock_client.write_register = AsyncMock(return_value=write_result)

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(MODBUS_CONFIG)
        result = await adapter.write("battery.soc_pct", 80.0)

    assert result is True


@pytest.mark.asyncio
async def test_write_fractional_scale() -> None:
    """Write with scale=0.1 correctly computes raw_value = value / 0.1."""
    adapter, mock_client = _create_adapter_with_mock()
    write_result = MagicMock()
    write_result.isError.return_value = False
    mock_client.write_register = AsyncMock(return_value=write_result)

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(MODBUS_CONFIG)
        # inverter.ac_input_power_w has scale=0.1, so writing 500W
        # should produce raw_value = 500 / 0.1 = 5000
        result = await adapter.write("inverter.ac_input_power_w", 500.0)

    assert result is True
    call_kwargs = mock_client.write_register.call_args.kwargs
    assert call_kwargs["value"] == 5000  # 500 / 0.1


@pytest.mark.asyncio
async def test_write_unknown_metric() -> None:
    """Writing unknown metric returns False."""
    adapter, mock_client = _create_adapter_with_mock()

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(MODBUS_CONFIG)
        result = await adapter.write("unknown.metric", 50.0)

    assert result is False


# ── Health Check Tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_check_connected() -> None:
    adapter, mock_client = _create_adapter_with_mock()

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(MODBUS_CONFIG)

    status = await adapter.health_check()
    assert status == AdapterStatus.CONNECTED


@pytest.mark.asyncio
async def test_health_check_disconnected() -> None:
    adapter = VictronModbusAdapter()
    status = await adapter.health_check()
    assert status == AdapterStatus.DISCONNECTED


@pytest.mark.asyncio
async def test_health_check_error_stale() -> None:
    """ERROR after 5 minutes without readings."""
    adapter, mock_client = _create_adapter_with_mock()

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(MODBUS_CONFIG)

    adapter._last_reading_at = time.time() - 400

    status = await adapter.health_check()
    assert status == AdapterStatus.ERROR


@pytest.mark.asyncio
async def test_health_check_error_count() -> None:
    """ERROR when error count exceeds 10."""
    adapter, mock_client = _create_adapter_with_mock()

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(MODBUS_CONFIG)

    adapter._error_count = 11

    status = await adapter.health_check()
    assert status == AdapterStatus.ERROR


# ── Subscribe Disconnect Detection ─────────────────────────────


@pytest.mark.asyncio
async def test_subscribe_raises_on_disconnect() -> None:
    """subscribe() raises RuntimeError after 3 consecutive empty polls with disconnected client."""
    adapter, mock_client = _create_adapter_with_mock()
    mock_client.read_holding_registers = AsyncMock(
        return_value=_make_register_result([100])
    )

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize({**MODBUS_CONFIG, "poll_interval_seconds": 0})

    # Simulate disconnect
    mock_client.connected = False

    callback = AsyncMock()
    with pytest.raises(RuntimeError, match="disconnected"):
        await adapter.subscribe(callback)


# ── Capabilities Tests ─────────────────────────────────────────


def test_capabilities() -> None:
    adapter = VictronModbusAdapter()
    adapter._config = MODBUS_CONFIG

    caps = adapter.capabilities()

    assert caps.can_read is True
    assert caps.can_write is True
    assert caps.can_subscribe is False
    assert "battery.soc_pct" in caps.supported_metrics
    assert "inverter.state" in caps.supported_metrics
    assert caps.poll_interval_seconds == 60
    assert caps.min_poll_interval_seconds == 5


# ── Shutdown Tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_shutdown() -> None:
    adapter, mock_client = _create_adapter_with_mock()

    with patch(
        "collector.adapters.victron_modbus.AsyncModbusTcpClient",
        return_value=mock_client,
    ):
        await adapter.initialize(MODBUS_CONFIG)
        await adapter.shutdown()

    mock_client.close.assert_called_once()
    assert adapter._status == AdapterStatus.DISCONNECTED
