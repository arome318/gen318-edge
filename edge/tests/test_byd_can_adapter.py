"""Tests for BYDCANAdapter (Doc 15 §7.4, F-HW-07)."""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import MagicMock, patch

import pytest

from collector.adapters.base import AdapterStatus
from collector.adapters.byd_can import (
    BYDCANAdapter,
    CELL_TEMP_BASE,
    CELL_VOLTAGE_BASE,
)

CAN_CONFIG: dict = {
    "interface": "can0",
    "bitrate": 500000,
    "asset_id": "asset-uuid-byd-battery",
}


def _make_can_message(arb_id: int, data: bytes) -> MagicMock:
    """Build a mock CAN message."""
    msg = MagicMock()
    msg.arbitration_id = arb_id
    msg.data = data
    return msg


# ── Initialize Tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_initialize_success() -> None:
    """CAN bus opens successfully."""
    adapter = BYDCANAdapter()

    mock_can = MagicMock()
    mock_bus = MagicMock()
    mock_can.Bus.return_value = mock_bus
    mock_reader = MagicMock()
    mock_can.AsyncBufferedReader.return_value = mock_reader
    mock_notifier = MagicMock()
    mock_can.Notifier.return_value = mock_notifier

    with patch.dict("sys.modules", {"can": mock_can}):
        result = await adapter.initialize(CAN_CONFIG)

    assert result is True
    assert adapter._status == AdapterStatus.CONNECTED
    mock_can.Bus.assert_called_once_with(
        interface="socketcan", channel="can0", bitrate=500000
    )


@pytest.mark.asyncio
async def test_initialize_failure() -> None:
    """CAN bus failure sets ERROR status."""
    adapter = BYDCANAdapter()

    mock_can = MagicMock()
    mock_can.Bus.side_effect = Exception("CAN interface not found")

    with patch.dict("sys.modules", {"can": mock_can}):
        result = await adapter.initialize(CAN_CONFIG)

    assert result is False
    assert adapter._status == AdapterStatus.ERROR


# ── Parse Message Tests ────────────────────────────────────────


def test_parse_0x355_soc_soh() -> None:
    """CAN ID 0x355: SoC and SoH parsed correctly."""
    adapter = BYDCANAdapter()
    adapter._asset_id = "asset-uuid-byd-battery"

    # SoC=85%, SoH=97%
    data = struct.pack("<HH", 85, 97) + b"\x00" * 4
    msg = _make_can_message(0x355, data)

    from datetime import UTC, datetime

    now = datetime.now(UTC)
    readings = adapter._parse_message(msg, now)

    assert len(readings) == 2

    soc = next(r for r in readings if r.metric_name == "battery.soc_pct")
    assert soc.value == 85.0
    assert soc.asset_id == "asset-uuid-byd-battery"

    soh = next(r for r in readings if r.metric_name == "battery.soh_pct")
    assert soh.value == 97.0


def test_parse_0x356_voltage_current_temp() -> None:
    """CAN ID 0x356: voltage, current, temperature parsed with correct scaling."""
    adapter = BYDCANAdapter()
    adapter._asset_id = "asset-uuid-byd-battery"

    # Voltage=5120 (51.20V), Current=-100 (-10.0A), Temp=250 (25.0°C)
    data = struct.pack("<Hhh", 5120, -100, 250) + b"\x00" * 2
    msg = _make_can_message(0x356, data)

    from datetime import UTC, datetime

    now = datetime.now(UTC)
    readings = adapter._parse_message(msg, now)

    assert len(readings) == 3

    voltage = next(r for r in readings if r.metric_name == "battery.voltage_v")
    assert abs(voltage.value - 51.20) < 0.01

    current = next(r for r in readings if r.metric_name == "battery.current_a")
    assert abs(current.value - (-10.0)) < 0.01

    temp = next(r for r in readings if r.metric_name == "battery.temperature_c")
    assert abs(temp.value - 25.0) < 0.01


def test_parse_0x351_charge_limits() -> None:
    """CAN ID 0x351: charge/discharge limits parsed."""
    adapter = BYDCANAdapter()
    adapter._asset_id = "asset-uuid-byd-battery"

    # ChargeV=560 (56.0V), ChargeCL=250 (25.0A), DischCL=250 (25.0A)
    data = struct.pack("<HHH", 560, 250, 250) + b"\x00" * 2
    msg = _make_can_message(0x351, data)

    from datetime import UTC, datetime

    now = datetime.now(UTC)
    readings = adapter._parse_message(msg, now)

    assert len(readings) == 3
    cv = next(r for r in readings if r.metric_name == "battery.charge_voltage_v")
    assert abs(cv.value - 56.0) < 0.01


def test_parse_cell_voltages() -> None:
    """Cell voltage messages 0x370-0x373 populate cell voltage array."""
    adapter = BYDCANAdapter()
    adapter._asset_id = "asset-uuid-byd-battery"

    from datetime import UTC, datetime

    now = datetime.now(UTC)

    # Module 0 (0x370): 4 cells at 3.2V, 3.3V, 3.1V, 3.4V (in mV)
    data0 = struct.pack("<HHHH", 3200, 3300, 3100, 3400)
    msg0 = _make_can_message(CELL_VOLTAGE_BASE, data0)
    adapter._parse_message(msg0, now)

    # Module 1 (0x371): 4 cells at 3.25V, 3.35V, 3.15V, 3.45V
    data1 = struct.pack("<HHHH", 3250, 3350, 3150, 3450)
    msg1 = _make_can_message(CELL_VOLTAGE_BASE + 1, data1)
    adapter._parse_message(msg1, now)

    # Check cell voltages accumulated
    assert len(adapter._cell_voltages) == 8
    assert abs(adapter._cell_voltages[0] - 3.2) < 0.001
    assert abs(adapter._cell_voltages[4] - 3.25) < 0.001


def test_parse_cell_temps() -> None:
    """Cell temperature messages 0x380-0x383 populate temp array."""
    adapter = BYDCANAdapter()
    adapter._asset_id = "asset-uuid-byd-battery"

    from datetime import UTC, datetime

    now = datetime.now(UTC)

    # Module 0 (0x380): temps in 0.1°C: 250, 260, 240, 255 → 25.0, 26.0, 24.0, 25.5
    data = struct.pack("<hhhh", 250, 260, 240, 255)
    msg = _make_can_message(CELL_TEMP_BASE, data)
    adapter._parse_message(msg, now)

    assert len(adapter._cell_temps) == 4
    assert abs(adapter._cell_temps[0] - 25.0) < 0.01
    assert abs(adapter._cell_temps[2] - 24.0) < 0.01


def test_parse_short_frame_handled_gracefully() -> None:
    """Truncated CAN frames don't crash the parser (caught by poll)."""
    adapter = BYDCANAdapter()
    adapter._asset_id = "asset-uuid-byd-battery"

    # A 0x356 message normally needs 6 bytes, but we send only 2
    short_data = b"\x00\x01"
    msg = _make_can_message(0x356, short_data)

    from datetime import UTC, datetime

    now = datetime.now(UTC)
    # _parse_message may raise struct.error, which poll() catches
    import struct

    try:
        adapter._parse_message(msg, now)
        # If it doesn't raise, it should return partial or empty
    except (struct.error, IndexError):
        pass  # Expected — poll() catches this


def test_parse_negative_temperature() -> None:
    """Negative temperatures parsed correctly (signed int16)."""
    adapter = BYDCANAdapter()
    adapter._asset_id = "asset-uuid-byd-battery"

    from datetime import UTC, datetime

    now = datetime.now(UTC)

    # Temperature = -50 → -5.0°C
    data = struct.pack("<hhhh", -50, -100, 0, 100)
    msg = _make_can_message(CELL_TEMP_BASE, data)
    adapter._parse_message(msg, now)

    assert abs(adapter._cell_temps[0] - (-5.0)) < 0.01
    assert abs(adapter._cell_temps[1] - (-10.0)) < 0.01


# ── Derive Cell Metrics Tests ──────────────────────────────────


def test_derive_cell_min_max_voltage() -> None:
    """Min/max cell voltage derived from accumulated data."""
    adapter = BYDCANAdapter()
    adapter._asset_id = "asset-uuid-byd-battery"
    adapter._cell_voltages = [3.2, 3.3, 3.1, 3.4, 3.25, 3.35, 3.15, 3.45]

    from datetime import UTC, datetime

    now = datetime.now(UTC)
    derived = adapter._derive_cell_metrics(now)

    # Should produce 4 readings (min/max voltage + min/max temp is 0 since no temps)
    voltage_derived = [r for r in derived if "voltage" in r.metric_name]
    assert len(voltage_derived) == 2

    min_v = next(r for r in derived if r.metric_name == "battery.cell_min_voltage_v")
    assert abs(min_v.value - 3.1) < 0.001

    max_v = next(r for r in derived if r.metric_name == "battery.cell_max_voltage_v")
    assert abs(max_v.value - 3.45) < 0.001


def test_derive_cell_min_max_temp() -> None:
    """Min/max cell temperature derived from accumulated data."""
    adapter = BYDCANAdapter()
    adapter._asset_id = "asset-uuid-byd-battery"
    adapter._cell_temps = [25.0, 26.0, 24.0, 25.5]

    from datetime import UTC, datetime

    now = datetime.now(UTC)
    derived = adapter._derive_cell_metrics(now)

    temp_derived = [r for r in derived if "temp" in r.metric_name]
    assert len(temp_derived) == 2

    min_t = next(r for r in derived if r.metric_name == "battery.cell_min_temp_c")
    assert abs(min_t.value - 24.0) < 0.01

    max_t = next(r for r in derived if r.metric_name == "battery.cell_max_temp_c")
    assert abs(max_t.value - 26.0) < 0.01


def test_derive_no_data_empty() -> None:
    """No derived metrics when no cell data accumulated."""
    adapter = BYDCANAdapter()
    adapter._asset_id = "asset-uuid-byd-battery"

    from datetime import UTC, datetime

    now = datetime.now(UTC)
    derived = adapter._derive_cell_metrics(now)

    assert derived == []


# ── Poll Integration Test ──────────────────────────────────────


@pytest.mark.asyncio
async def test_poll_with_messages() -> None:
    """Poll reads CAN messages via AsyncBufferedReader."""
    adapter = BYDCANAdapter()
    adapter._asset_id = "asset-uuid-byd-battery"

    # Build mock messages
    soc_data = struct.pack("<HH", 85, 97) + b"\x00" * 4
    soc_msg = _make_can_message(0x355, soc_data)

    voltage_data = struct.pack("<HHHH", 3200, 3300, 3100, 3400)
    voltage_msg = _make_can_message(CELL_VOLTAGE_BASE, voltage_data)

    messages = [soc_msg, voltage_msg]
    call_count = 0

    async def mock_get_message() -> MagicMock | None:
        nonlocal call_count
        if call_count < len(messages):
            msg = messages[call_count]
            call_count += 1
            return msg
        # After messages exhausted, wait for timeout
        await asyncio.sleep(1)
        return None

    adapter._reader = MagicMock()
    adapter._reader.get_message = mock_get_message
    adapter._bus = MagicMock()

    readings = await adapter.poll()

    # Should get SoC + SoH (from 0x355) + derived min/max voltage (from 0x370)
    assert len(readings) >= 2
    soc_readings = [r for r in readings if r.metric_name == "battery.soc_pct"]
    assert len(soc_readings) == 1
    assert soc_readings[0].value == 85.0


@pytest.mark.asyncio
async def test_poll_no_reader() -> None:
    """Returns empty if reader is not initialized."""
    adapter = BYDCANAdapter()
    readings = await adapter.poll()
    assert readings == []


# ── Health Check Tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_check_disconnected() -> None:
    adapter = BYDCANAdapter()
    status = await adapter.health_check()
    assert status == AdapterStatus.DISCONNECTED


@pytest.mark.asyncio
async def test_health_check_error_stale() -> None:
    """ERROR after 2 minutes without readings."""
    import time

    adapter = BYDCANAdapter()
    adapter._bus = MagicMock()
    adapter._last_reading_at = time.time() - 150

    status = await adapter.health_check()
    assert status == AdapterStatus.ERROR


@pytest.mark.asyncio
async def test_health_check_connected() -> None:
    adapter = BYDCANAdapter()
    adapter._bus = MagicMock()
    adapter._last_reading_at = 0.0  # never read but just init

    status = await adapter.health_check()
    assert status == AdapterStatus.CONNECTED


# ── Capabilities / Write / Shutdown ────────────────────────────


def test_capabilities() -> None:
    adapter = BYDCANAdapter()
    caps = adapter.capabilities()

    assert caps.can_read is True
    assert caps.can_write is False
    assert caps.can_subscribe is True
    assert "battery.soc_pct" in caps.supported_metrics
    assert "battery.cell_min_voltage_v" in caps.supported_metrics
    assert caps.poll_interval_seconds == 1


@pytest.mark.asyncio
async def test_write_returns_false() -> None:
    adapter = BYDCANAdapter()
    assert await adapter.write("battery.soc_pct", 50.0) is False


@pytest.mark.asyncio
async def test_shutdown() -> None:
    adapter = BYDCANAdapter()
    adapter._bus = MagicMock()
    adapter._notifier = MagicMock()

    await adapter.shutdown()

    adapter._notifier.stop.assert_called_once()
    adapter._bus.shutdown.assert_called_once()
    assert adapter._status == AdapterStatus.DISCONNECTED
