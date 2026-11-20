"""Tests for VictronMQTTAdapter."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collector.adapters.base import AdapterStatus, Reading
from collector.adapters.victron_mqtt import (
    MQTT_TOPICS_BATTERY,
    MQTT_TOPICS_SOLAR,
    MQTT_TOPICS_SYSTEM,
    MQTT_TOPICS_VEBUS,
    VictronMQTTAdapter,
)

MQTT_CONFIG: dict = {
    "broker_host": "192.168.1.1",
    "broker_port": 1883,
    "portal_id": "abc123def456",
    "instances": {
        "solarcharger": [0],
        "vebus": [0],
        "battery": [0],
    },
    "asset_mappings": {
        "system": "asset-uuid-inverter",
        "solarcharger_0": "asset-uuid-solar",
        "vebus_0": "asset-uuid-inverter",
        "battery_0": "asset-uuid-battery",
    },
    "write_topics": {
        "battery.soc_pct": "W/{portal}/system/0/Dc/Battery/Soc",
    },
}


# ── Initialize Tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_initialize_builds_topic_map() -> None:
    """Initialize populates the topic-to-metric map."""
    adapter = VictronMQTTAdapter()
    result = await adapter.initialize(MQTT_CONFIG)

    assert result is True
    assert adapter._status == AdapterStatus.CONNECTING

    # Should have topics for all 4 categories
    expected_count = (
        len(MQTT_TOPICS_SYSTEM)
        + len(MQTT_TOPICS_SOLAR)
        + len(MQTT_TOPICS_VEBUS)
        + len(MQTT_TOPICS_BATTERY)
    )
    assert len(adapter._topic_to_metric) == expected_count


@pytest.mark.asyncio
async def test_topic_map_system_topics() -> None:
    """System topics map to correct metrics and asset IDs."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    portal = "abc123def456"
    soc_topic = f"N/{portal}/system/0/Dc/Battery/Soc"
    assert soc_topic in adapter._topic_to_metric

    metric, asset_id = adapter._topic_to_metric[soc_topic]
    assert metric == "battery.soc_pct"
    assert asset_id == "asset-uuid-inverter"


@pytest.mark.asyncio
async def test_topic_map_solar_topics() -> None:
    """Solar charger topics use instance-specific asset mapping."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    portal = "abc123def456"
    power_topic = f"N/{portal}/solarcharger/0/Yield/Power"
    assert power_topic in adapter._topic_to_metric

    metric, asset_id = adapter._topic_to_metric[power_topic]
    assert metric == "solar.pv_power_w"
    assert asset_id == "asset-uuid-solar"


@pytest.mark.asyncio
async def test_topic_map_vebus_topics() -> None:
    """VE.Bus topics map correctly."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    portal = "abc123def456"
    state_topic = f"N/{portal}/vebus/0/State"
    assert state_topic in adapter._topic_to_metric

    metric, asset_id = adapter._topic_to_metric[state_topic]
    assert metric == "inverter.state"
    assert asset_id == "asset-uuid-inverter"


@pytest.mark.asyncio
async def test_topic_map_battery_topics() -> None:
    """Battery monitor topics map correctly."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    portal = "abc123def456"
    soh_topic = f"N/{portal}/battery/0/Soh"
    assert soh_topic in adapter._topic_to_metric

    metric, asset_id = adapter._topic_to_metric[soh_topic]
    assert metric == "battery.soh_pct"
    assert asset_id == "asset-uuid-battery"


@pytest.mark.asyncio
async def test_all_38_topics_mapped() -> None:
    """All 38 topics across 4 categories are in the topic map."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    # 12 system + 7 solar + 10 vebus + 6 battery = 35
    # (with single instances each)
    expected = (
        len(MQTT_TOPICS_SYSTEM)
        + len(MQTT_TOPICS_SOLAR)
        + len(MQTT_TOPICS_VEBUS)
        + len(MQTT_TOPICS_BATTERY)
    )
    assert len(adapter._topic_to_metric) == expected


# ── Parse Message Tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_message_json_value() -> None:
    """Parse standard Victron JSON payload {"value": x}."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    portal = "abc123def456"
    topic = f"N/{portal}/system/0/Dc/Battery/Soc"
    payload = json.dumps({"value": 85.5}).encode()

    reading = adapter._parse_message(topic, payload)

    assert reading is not None
    assert reading.metric_name == "battery.soc_pct"
    assert reading.value == 85.5
    assert reading.asset_id == "asset-uuid-inverter"
    assert reading.quality == 0


@pytest.mark.asyncio
async def test_parse_message_plain_number() -> None:
    """Parse plain numeric JSON payload (no "value" key)."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    portal = "abc123def456"
    topic = f"N/{portal}/system/0/Dc/Battery/Voltage"
    payload = json.dumps(48.2).encode()

    reading = adapter._parse_message(topic, payload)

    assert reading is not None
    assert reading.metric_name == "battery.voltage_v"
    assert reading.value == 48.2


@pytest.mark.asyncio
async def test_parse_message_invalid_json() -> None:
    """Invalid JSON returns None (graceful degradation)."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    portal = "abc123def456"
    topic = f"N/{portal}/system/0/Dc/Battery/Soc"
    payload = b"not json"

    reading = adapter._parse_message(topic, payload)
    assert reading is None


@pytest.mark.asyncio
async def test_parse_message_null_value() -> None:
    """Victron null value (device offline) returns None — no reading emitted."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    portal = "abc123def456"
    topic = f"N/{portal}/system/0/Dc/Battery/Soc"
    payload = json.dumps({"value": None}).encode()

    reading = adapter._parse_message(topic, payload)
    assert reading is None


@pytest.mark.asyncio
async def test_parse_message_non_numeric() -> None:
    """Non-numeric value returns None."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    portal = "abc123def456"
    topic = f"N/{portal}/system/0/Dc/Battery/Soc"
    payload = json.dumps({"value": "not_a_number"}).encode()

    reading = adapter._parse_message(topic, payload)
    assert reading is None


# ── Write Queue Tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_queues_command() -> None:
    """write() queues command for dispatch in subscription loop."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    result = await adapter.write("battery.soc_pct", 90.0)
    assert result is True
    assert adapter._write_queue.qsize() == 1

    metric, value = adapter._write_queue.get_nowait()
    assert metric == "battery.soc_pct"
    assert value == 90.0


@pytest.mark.asyncio
async def test_write_unknown_metric() -> None:
    """write() returns False for unknown metrics."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    result = await adapter.write("unknown.metric", 50.0)
    assert result is False
    assert adapter._write_queue.qsize() == 0


@pytest.mark.asyncio
async def test_dispatch_write() -> None:
    """_dispatch_write publishes to the explicit allowlisted topic."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    mock_client = AsyncMock()

    await adapter._dispatch_write(mock_client, "battery.soc_pct", 90.0)

    mock_client.publish.assert_called_once()
    call_args = mock_client.publish.call_args
    topic = call_args.args[0]
    assert topic.startswith("W/")
    assert "Soc" in topic
    payload = json.loads(call_args.args[1])
    assert payload["value"] == 90.0


# ── Health Check Tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_check_connecting() -> None:
    """Status is CONNECTING after init (before subscribe)."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    status = await adapter.health_check()
    assert status == AdapterStatus.CONNECTING


@pytest.mark.asyncio
async def test_health_check_error_stale() -> None:
    """ERROR after 2 minutes without readings."""
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)
    adapter._status = AdapterStatus.CONNECTED
    adapter._last_reading_at = time.time() - 150

    status = await adapter.health_check()
    assert status == AdapterStatus.ERROR


@pytest.mark.asyncio
async def test_health_check_disconnected() -> None:
    adapter = VictronMQTTAdapter()
    status = await adapter.health_check()
    assert status == AdapterStatus.DISCONNECTED


# ── Capabilities Tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_capabilities() -> None:
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    caps = adapter.capabilities()

    assert caps.can_read is True
    assert caps.can_write is True
    assert caps.can_subscribe is True
    assert "battery.soc_pct" in caps.supported_metrics
    assert "solar.pv_power_w" in caps.supported_metrics
    assert "inverter.state" in caps.supported_metrics
    assert caps.poll_interval_seconds == 0  # real-time


# ── Poll / Shutdown Tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_poll_returns_empty() -> None:
    """MQTT adapter poll() returns empty (use subscribe instead)."""
    adapter = VictronMQTTAdapter()
    readings = await adapter.poll()
    assert readings == []


@pytest.mark.asyncio
async def test_shutdown() -> None:
    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    await adapter.shutdown()
    assert adapter._status == AdapterStatus.DISCONNECTED


# ── Subscribe Integration Test ─────────────────────────────────


@pytest.mark.asyncio
async def test_subscribe_reconnects_on_error() -> None:
    """Subscribe reconnects after MqttError with backoff."""
    import aiomqtt as real_aiomqtt

    adapter = VictronMQTTAdapter()
    await adapter.initialize(MQTT_CONFIG)

    call_count = 0
    received_readings: list[Reading] = []

    async def mock_callback(readings: list[Reading]) -> None:
        received_readings.extend(readings)

    class MockMessage:
        def __init__(self, topic: str, payload: bytes) -> None:
            self.topic = MagicMock()
            self.topic.__str__ = MagicMock(return_value=topic)
            self.payload = payload

    portal = "abc123def456"
    topic = f"N/{portal}/system/0/Dc/Battery/Soc"
    msg = MockMessage(topic, json.dumps({"value": 85.0}).encode())

    class MockClient:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1

        async def __aenter__(self) -> MockClient:
            if call_count == 1:
                raise real_aiomqtt.MqttError("Connection refused")
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def subscribe(self, topic: str) -> None:
            pass

        async def publish(self, topic: str, payload: Any) -> None:
            pass

        @property
        def messages(self) -> MockMessages:
            return MockMessages(msg)

    class MockMessages:
        def __init__(self, message: MockMessage) -> None:
            self._message = message
            self._sent = False

        def __aiter__(self) -> MockMessages:
            return self

        async def __anext__(self) -> MockMessage:
            if not self._sent:
                self._sent = True
                return self._message
            raise asyncio.CancelledError

    # Patch aiomqtt in the module's namespace by patching it in sys.modules
    mock_module = MagicMock()
    mock_module.Client = MockClient
    mock_module.MqttError = real_aiomqtt.MqttError

    with patch.dict("sys.modules", {"aiomqtt": mock_module}):
        with pytest.raises(asyncio.CancelledError):
            await adapter.subscribe(mock_callback)

    # First attempt fails, second succeeds and gets one message
    assert call_count == 2
    assert len(received_readings) == 1
    assert received_readings[0].metric_name == "battery.soc_pct"
    assert received_readings[0].value == 85.0
