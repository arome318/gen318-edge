"""Tests for GenericModbusAdapter."""

from __future__ import annotations

import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collector.adapters.generic_modbus import DEVICE_KWARG, GenericModbusAdapter


# ── Fixtures ────────────────────────────────────────────────────


def _make_register_result(registers: list[int]) -> MagicMock:
    """Build a mock Modbus response with given register values."""
    result = MagicMock()
    result.isError.return_value = False
    result.registers = registers
    return result


def _make_error_result() -> MagicMock:
    result = MagicMock()
    result.isError.return_value = True
    return result


TCP_CONFIG: dict = {
    "protocol": "tcp",
    "host": "192.168.1.100",
    "port": 502,
    "timeout_ms": 3000,
    "retries": 3,
    "unit_id": 1,
    "poll_interval_seconds": 60,
    "asset_id": "asset-uuid-genset",
    "registers": [
        {
            "metric": "genset.power_w",
            "address": 100,
            "type": "uint16",
            "scale": 1.0,
            "function_code": 4,
        },
        {
            "metric": "genset.runtime_hours",
            "address": 101,
            "type": "uint32",
            "scale": 0.1,
            "function_code": 4,
        },
        {
            "metric": "genset.fuel_level_pct",
            "address": 103,
            "type": "uint16",
            "scale": 0.1,
            "function_code": 4,
        },
        {
            "metric": "genset.status",
            "address": 104,
            "type": "uint16",
            "scale": 1,
            "function_code": 4,
        },
    ],
}

RTU_CONFIG: dict = {
    "protocol": "rtu",
    "port": "/dev/ttyUSB0",
    "baudrate": 9600,
    "timeout_ms": 3000,
    "retries": 3,
    "unit_id": 1,
    "poll_interval_seconds": 60,
    "asset_id": "asset-uuid-rtu",
    "registers": [
        {
            "metric": "meter.power_w",
            "address": 0,
            "type": "int16",
            "scale": 1.0,
            "function_code": 3,
        },
    ],
}


# ── Initialize Tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_init_tcp() -> None:
    """TCP mode creates AsyncModbusTcpClient and connects."""
    adapter = GenericModbusAdapter()

    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.connect = AsyncMock(return_value=True)
        mock_instance.connected = True
        MockClient.return_value = mock_instance

        result = await adapter.initialize(TCP_CONFIG)

    assert result is True
    MockClient.assert_called_once()
    call_kwargs = MockClient.call_args.kwargs
    assert call_kwargs["host"] == "192.168.1.100"
    assert call_kwargs["port"] == 502
    assert call_kwargs["reconnect_delay"] == 1
    assert call_kwargs["reconnect_delay_max"] == 60


@pytest.mark.asyncio
async def test_init_rtu() -> None:
    """RTU mode creates AsyncModbusSerialClient with correct serial params."""
    adapter = GenericModbusAdapter()

    with patch(
        "collector.adapters.generic_modbus.AsyncModbusSerialClient"
    ) as MockClient:
        mock_instance = AsyncMock()
        mock_instance.connect = AsyncMock(return_value=True)
        mock_instance.connected = True
        MockClient.return_value = mock_instance

        result = await adapter.initialize(RTU_CONFIG)

    assert result is True
    call_kwargs = MockClient.call_args.kwargs
    assert call_kwargs["port"] == "/dev/ttyUSB0"
    assert call_kwargs["baudrate"] == 9600
    assert call_kwargs["parity"] == "N"
    assert call_kwargs["stopbits"] == 1


@pytest.mark.asyncio
async def test_init_unsupported_protocol() -> None:
    """Unsupported protocol returns False."""
    adapter = GenericModbusAdapter()
    config = {**TCP_CONFIG, "protocol": "profinet"}
    result = await adapter.initialize(config)
    assert result is False


@pytest.mark.asyncio
async def test_init_connection_failure() -> None:
    """Connection failure sets ERROR status."""
    adapter = GenericModbusAdapter()

    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.connect = AsyncMock(return_value=False)
        MockClient.return_value = mock_instance

        result = await adapter.initialize(TCP_CONFIG)

    assert result is False


# ── Poll Tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_poll_reads_all_registers() -> None:
    """Poll reads all configured registers and returns readings."""
    adapter = GenericModbusAdapter()

    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.connected = True
        # Function code 4 → read_input_registers
        mock_client.read_input_registers = AsyncMock(
            side_effect=[
                _make_register_result([5000]),  # power_w: 5000 * 1.0 = 5000.0
                _make_register_result(
                    [0, 36000]
                ),  # runtime: (0 << 16 | 36000) * 0.1 = 3600.0
                _make_register_result([750]),  # fuel_pct: 750 * 0.1 = 75.0
                _make_register_result([1]),  # status: 1 * 1 = 1.0
            ]
        )
        MockClient.return_value = mock_client
        await adapter.initialize(TCP_CONFIG)

        readings = await adapter.poll()

    assert len(readings) == 4
    assert readings[0].metric_name == "genset.power_w"
    assert readings[0].value == 5000.0
    assert readings[0].asset_id == "asset-uuid-genset"

    assert readings[1].metric_name == "genset.runtime_hours"
    assert readings[1].value == 3600.0

    assert readings[2].metric_name == "genset.fuel_level_pct"
    assert readings[2].value == 75.0

    assert readings[3].metric_name == "genset.status"
    assert readings[3].value == 1.0


@pytest.mark.asyncio
async def test_poll_function_code_3() -> None:
    """Function code 3 uses read_holding_registers."""
    adapter = GenericModbusAdapter()
    config = {
        **TCP_CONFIG,
        "registers": [
            {
                "metric": "test.value",
                "address": 10,
                "type": "uint16",
                "scale": 1.0,
                "function_code": 3,
            },
        ],
    }

    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.connected = True
        mock_client.read_holding_registers = AsyncMock(
            return_value=_make_register_result([42])
        )
        MockClient.return_value = mock_client
        await adapter.initialize(config)

        readings = await adapter.poll()

    assert len(readings) == 1
    assert readings[0].value == 42.0
    # pymodbus renamed this keyword from `slave` to `device_id` and removed
    # the old name in 3.14. The adapter resolves it from the installed
    # signature, so the assertion follows rather than pinning a name that
    # depends on which wheel the node happened to build against.
    mock_client.read_holding_registers.assert_called_once_with(
        address=10, count=1, **{DEVICE_KWARG: 1}
    )


@pytest.mark.asyncio
async def test_poll_int16_signed_conversion() -> None:
    """int16 register value 65535 reads as -1."""
    adapter = GenericModbusAdapter()
    config = {
        **TCP_CONFIG,
        "registers": [
            {
                "metric": "test.signed",
                "address": 0,
                "type": "int16",
                "scale": 1.0,
                "function_code": 3,
            },
        ],
    }

    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.connected = True
        mock_client.read_holding_registers = AsyncMock(
            return_value=_make_register_result([65535])
        )
        MockClient.return_value = mock_client
        await adapter.initialize(config)

        readings = await adapter.poll()

    assert len(readings) == 1
    assert readings[0].value == -1.0


@pytest.mark.asyncio
async def test_poll_int32_signed() -> None:
    """int32 type correctly handles signed 32-bit values."""
    adapter = GenericModbusAdapter()
    config = {
        **TCP_CONFIG,
        "registers": [
            {
                "metric": "test.int32",
                "address": 0,
                "type": "int32",
                "scale": 1.0,
                "function_code": 3,
            },
        ],
    }

    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.connected = True
        # 0xFFFF_FFFF = -1 as signed int32
        mock_client.read_holding_registers = AsyncMock(
            return_value=_make_register_result([0xFFFF, 0xFFFF])
        )
        MockClient.return_value = mock_client
        await adapter.initialize(config)

        readings = await adapter.poll()

    assert len(readings) == 1
    assert readings[0].value == -1.0


@pytest.mark.asyncio
async def test_poll_float32() -> None:
    """float32 type correctly unpacks IEEE 754 float."""
    adapter = GenericModbusAdapter()
    expected_value = 3.14
    packed = struct.pack(">f", expected_value)
    high, low = struct.unpack(">HH", packed)

    config = {
        **TCP_CONFIG,
        "registers": [
            {
                "metric": "test.float",
                "address": 0,
                "type": "float32",
                "scale": 1.0,
                "function_code": 3,
            },
        ],
    }

    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.connected = True
        mock_client.read_holding_registers = AsyncMock(
            return_value=_make_register_result([high, low])
        )
        MockClient.return_value = mock_client
        await adapter.initialize(config)

        readings = await adapter.poll()

    assert len(readings) == 1
    assert abs(readings[0].value - expected_value) < 0.01


@pytest.mark.asyncio
async def test_poll_scale_factor() -> None:
    """Scale factor is applied correctly."""
    adapter = GenericModbusAdapter()
    config = {
        **TCP_CONFIG,
        "registers": [
            {
                "metric": "test.scaled",
                "address": 0,
                "type": "uint16",
                "scale": 0.1,
                "function_code": 3,
            },
        ],
    }

    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.connected = True
        mock_client.read_holding_registers = AsyncMock(
            return_value=_make_register_result([1234])
        )
        MockClient.return_value = mock_client
        await adapter.initialize(config)

        readings = await adapter.poll()

    assert len(readings) == 1
    assert abs(readings[0].value - 123.4) < 0.001


@pytest.mark.asyncio
async def test_poll_register_error_skipped() -> None:
    """Registers that return errors are skipped."""
    adapter = GenericModbusAdapter()

    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.connected = True
        mock_client.read_input_registers = AsyncMock(
            side_effect=[
                _make_error_result(),  # power_w fails
                _make_register_result([0, 36000]),  # runtime succeeds
                _make_error_result(),  # fuel fails
                _make_register_result([1]),  # status succeeds
            ]
        )
        MockClient.return_value = mock_client
        await adapter.initialize(TCP_CONFIG)

        readings = await adapter.poll()

    assert len(readings) == 2
    assert readings[0].metric_name == "genset.runtime_hours"
    assert readings[1].metric_name == "genset.status"


@pytest.mark.asyncio
async def test_poll_disconnected_returns_empty() -> None:
    """Returns empty list when client is not connected."""
    adapter = GenericModbusAdapter()

    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.connected = False  # simulate disconnection
        MockClient.return_value = mock_client
        await adapter.initialize(TCP_CONFIG)

        readings = await adapter.poll()

    assert readings == []


# ── Health Check Tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_check_connected() -> None:
    """Health check returns CONNECTED when client is healthy."""
    adapter = GenericModbusAdapter()

    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.connected = True
        MockClient.return_value = mock_client
        await adapter.initialize(TCP_CONFIG)
        adapter._last_reading_at = 0.0  # never read, but just initialized

        status = await adapter.health_check()

    assert status == adapter._status.CONNECTED


@pytest.mark.asyncio
async def test_health_check_error_no_readings() -> None:
    """Health check returns ERROR after 5 minutes without readings."""
    adapter = GenericModbusAdapter()

    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.connected = True
        MockClient.return_value = mock_client
        await adapter.initialize(TCP_CONFIG)

    import time

    adapter._last_reading_at = time.time() - 400  # >5 min ago

    status = await adapter.health_check()
    from collector.adapters.base import AdapterStatus

    assert status == AdapterStatus.ERROR


@pytest.mark.asyncio
async def test_health_check_disconnected() -> None:
    """Health check returns DISCONNECTED when client is gone."""
    adapter = GenericModbusAdapter()
    status = await adapter.health_check()
    from collector.adapters.base import AdapterStatus

    assert status == AdapterStatus.DISCONNECTED


# ── Capabilities Tests ─────────────────────────────────────────


def test_capabilities() -> None:
    """Capabilities reflect configured registers."""
    adapter = GenericModbusAdapter()
    adapter._config = TCP_CONFIG
    adapter._registers = TCP_CONFIG["registers"]

    caps = adapter.capabilities()

    assert caps.can_read is True
    assert caps.can_write is False
    assert caps.can_subscribe is False
    assert set(caps.supported_metrics) == {
        "genset.power_w",
        "genset.runtime_hours",
        "genset.fuel_level_pct",
        "genset.status",
    }
    assert caps.poll_interval_seconds == 60
    assert caps.min_poll_interval_seconds == 5


# ── Shutdown Tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_shutdown() -> None:
    """Shutdown closes the client."""
    adapter = GenericModbusAdapter()

    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.connected = True
        mock_client.close = MagicMock()
        MockClient.return_value = mock_client
        await adapter.initialize(TCP_CONFIG)

        await adapter.shutdown()

    mock_client.close.assert_called_once()
    from collector.adapters.base import AdapterStatus

    assert adapter._status == AdapterStatus.DISCONNECTED


# ── Write Tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_returns_false() -> None:
    """Generic adapter does not support writes."""
    adapter = GenericModbusAdapter()
    result = await adapter.write("genset.power_w", 1000.0)
    assert result is False
