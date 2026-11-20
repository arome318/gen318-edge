"""Integration tests for CollectorService.

Tests the collector's adapter orchestration, ManagedAdapter tracking,
AdapterSupervisor restart behavior, and readings flow to SQLite + MQTT.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from collector.adapters.base import (
    AdapterCapabilities,
    AdapterStatus,
    HardwareAdapter,
    Reading,
)
from collector.main import AdapterSupervisor, CollectorService, ManagedAdapter


# ── Mock Adapter ───────────────────────────────────────────────


class MockPollAdapter(HardwareAdapter):
    """Mock poll-based adapter for testing."""

    def __init__(self) -> None:
        self._readings: list[Reading] = []
        self._initialized = False
        self._shutdown_called = False

    async def initialize(self, config: dict) -> bool:
        self._initialized = True
        return True

    async def poll(self) -> list[Reading]:
        return self._readings

    async def subscribe(self, callback) -> bool:
        return False

    async def write(self, metric_name: str, value: float) -> bool:
        return False

    async def health_check(self) -> AdapterStatus:
        return AdapterStatus.CONNECTED

    async def shutdown(self) -> None:
        self._shutdown_called = True

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            can_read=True,
            can_write=False,
            can_subscribe=False,
            poll_interval_seconds=1,
            min_poll_interval_seconds=1,
        )


class MockSubscribeAdapter(HardwareAdapter):
    """Mock subscribe-based adapter for testing."""

    def __init__(self) -> None:
        self._initialized = False
        self._shutdown_called = False
        self._readings_to_send: list[Reading] = []

    async def initialize(self, config: dict) -> bool:
        self._initialized = True
        return True

    async def poll(self) -> list[Reading]:
        return []

    async def subscribe(self, callback) -> bool:
        if self._readings_to_send:
            await callback(self._readings_to_send)
        return True

    async def write(self, metric_name: str, value: float) -> bool:
        return False

    async def health_check(self) -> AdapterStatus:
        return AdapterStatus.CONNECTED

    async def shutdown(self) -> None:
        self._shutdown_called = True

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            can_read=True,
            can_write=False,
            can_subscribe=True,
            poll_interval_seconds=0,
            min_poll_interval_seconds=0,
        )


# ── ManagedAdapter Tests ───────────────────────────────────────


def test_managed_adapter_record_readings() -> None:
    """record_readings updates counters and resets errors."""
    managed = ManagedAdapter(
        name="test",
        adapter_type="mock",
        config={},
        error_count=5,
    )

    managed.record_readings(10)

    assert managed.reading_count == 10
    assert managed.error_count == 0
    assert managed.status == AdapterStatus.CONNECTED
    assert managed.last_reading_at > 0


def test_managed_adapter_record_error() -> None:
    """record_error increments count and sets ERROR after threshold."""
    managed = ManagedAdapter(name="test", adapter_type="mock", config={})

    for _ in range(10):
        managed.record_error()
    assert managed.status != AdapterStatus.ERROR

    managed.record_error()  # 11th error
    assert managed.status == AdapterStatus.ERROR


def test_managed_adapter_to_status_dict() -> None:
    """Status dict includes all expected fields."""
    managed = ManagedAdapter(
        name="victron_mqtt",
        adapter_type="victron_mqtt",
        config={},
        status=AdapterStatus.CONNECTED,
        reading_count=100,
    )

    status = managed.to_status_dict()

    assert status["name"] == "victron_mqtt"
    assert status["type"] == "victron_mqtt"
    assert status["status"] == "connected"
    assert status["reading_count"] == 100


# ── AdapterSupervisor Tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_supervisor_starts_tasks() -> None:
    """Supervisor starts all registered tasks."""
    supervisor = AdapterSupervisor(max_restarts=3)
    started = []

    async def task_a() -> None:
        started.append("a")
        await asyncio.sleep(10)

    async def task_b() -> None:
        started.append("b")
        await asyncio.sleep(10)

    supervisor.add("a", task_a)
    supervisor.add("b", task_b)

    # Run briefly then stop
    run_task = asyncio.create_task(supervisor.run())
    await asyncio.sleep(0.1)
    await supervisor.stop()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

    assert "a" in started
    assert "b" in started


@pytest.mark.asyncio
async def test_supervisor_restarts_failed_task() -> None:
    """Supervisor restarts a task that raises an exception."""
    supervisor = AdapterSupervisor(max_restarts=3)
    call_count = 0

    async def failing_task() -> None:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise RuntimeError("simulated failure")
        await asyncio.sleep(10)  # Stay alive on 3rd attempt

    supervisor.add("failer", failing_task)

    run_task = asyncio.create_task(supervisor.run())
    await asyncio.sleep(8)  # Enough time for 2 restarts with backoff
    await supervisor.stop()
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass

    assert call_count >= 2  # At least 2 attempts


# ── CollectorService Tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_on_readings_writes_to_db_and_mqtt(edge_db, mock_mqtt) -> None:
    """Readings are stored in SQLite and published to MQTT."""
    from common.config import EdgeConfig

    config = EdgeConfig(
        device_id="test-edge-001",
        cloud_api_url="https://api.gen318.io",
        device_cert_path="/tmp/fake.pem",
        device_key_path="/tmp/fake.key",
        ca_cert_path="/tmp/fake-ca.pem",
        cert_pin_sha256="",
        sqlite_path="/tmp/test.db",
        sqlite_encryption_key="",
    )

    service = CollectorService(config)
    service.db = edge_db
    service.mqtt = mock_mqtt

    managed = ManagedAdapter(
        name="test_adapter",
        adapter_type="generic_modbus",
        config={},
    )

    readings = [
        Reading(
            metric_name="battery.soc_pct",
            value=85.0,
            timestamp=datetime.now(UTC),
            quality=0,
            raw_register="test:0",
            asset_id="asset-uuid-001",
        ),
        Reading(
            metric_name="solar.pv_power_w",
            value=3500.0,
            timestamp=datetime.now(UTC),
            quality=0,
            raw_register="test:1",
            asset_id="asset-uuid-002",
        ),
    ]

    await service._on_readings(readings, managed)

    # Verify SQLite writes
    stored = edge_db.get_latest_readings()
    assert len(stored) == 2
    soc = next(r for r in stored if r["metric_name"] == "battery.soc_pct")
    assert soc["value"] == 85.0
    assert soc["asset_id"] == "asset-uuid-001"

    # Verify MQTT publishes
    assert mock_mqtt.publish.call_count == 2

    # Verify managed adapter tracking
    assert managed.reading_count == 2
    assert managed.error_count == 0
    assert managed.last_reading_at > 0


@pytest.mark.asyncio
async def test_pq_interruption_escalation_enqueues_priority_zero(
    edge_db, mock_mqtt
) -> None:
    """Fix round 1 / IMPORTANT-2, end-to-end through the real collector +
    SQLite sync queue (not just the detector in isolation, per
    test_pq_detector.py::test_worsening_undervoltage_escalates_to_
    interruption_priority_zero): 220V -> 200Vx2 (undervoltage opens
    silently) -> 15V (escalates: undervoltage force-closes onto the P1
    queue) -> 15V (interruption opens onto the P0 queue)."""
    from common.config import EdgeConfig

    config = EdgeConfig(
        device_id="test-edge-001",
        cloud_api_url="https://api.gen318.io",
        device_cert_path="/tmp/fake.pem",
        device_key_path="/tmp/fake.key",
        ca_cert_path="/tmp/fake-ca.pem",
        cert_pin_sha256="",
        sqlite_path="/tmp/test.db",
        sqlite_encryption_key="",
    )
    service = CollectorService(config)
    service.db = edge_db
    service.mqtt = mock_mqtt
    managed = ManagedAdapter(name="meter", adapter_type="generic_modbus", config={})

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    asset_id = "asset-uuid-001"

    async def feed(value: float, offset_seconds: int) -> None:
        readings = [
            Reading(
                metric_name="meter.voltage_v",
                value=value,
                timestamp=t0 + timedelta(seconds=offset_seconds),
                asset_id=asset_id,
            )
        ]
        await service._on_readings(readings, managed)

    await feed(220.0, 0)
    await feed(200.0, 30)
    await feed(200.0, 90)  # undervoltage opens silently (no queue item yet)
    await feed(15.0, 120)  # escalation: undervoltage force-closes -> P1
    await feed(15.0, 180)  # interruption opens -> P0

    queued_p0 = edge_db.get_queue_items(priority=0)
    assert len(queued_p0) == 1
    assert queued_p0[0]["payload"]["type"] == "pq_event"
    assert queued_p0[0]["payload"]["event_type"] == "supply_interruption"
    assert queued_p0[0]["payload"]["severity"] == "critical"
    assert queued_p0[0]["payload"]["ended_at"] is None

    queued_p1 = edge_db.get_queue_items(priority=1)
    assert any(item["payload"]["event_type"] == "undervoltage" for item in queued_p1)


@pytest.mark.asyncio
async def test_on_readings_empty_noop(edge_db, mock_mqtt) -> None:
    """Empty readings list is a no-op."""
    from common.config import EdgeConfig

    config = EdgeConfig(
        device_id="test-edge-001",
        cloud_api_url="https://api.gen318.io",
        device_cert_path="/tmp/fake.pem",
        device_key_path="/tmp/fake.key",
        ca_cert_path="/tmp/fake-ca.pem",
        cert_pin_sha256="",
        sqlite_path="/tmp/test.db",
        sqlite_encryption_key="",
    )

    service = CollectorService(config)
    service.db = edge_db
    service.mqtt = mock_mqtt

    managed = ManagedAdapter(name="test", adapter_type="mock", config={})

    await service._on_readings([], managed)

    assert mock_mqtt.publish.call_count == 0
    assert managed.reading_count == 0


def test_load_adapter_config_missing(edge_config) -> None:
    """Missing config file returns empty adapter list."""
    service = CollectorService(edge_config)
    service.config = edge_config

    adapters = service._load_adapter_config()
    assert adapters == []


@pytest.mark.asyncio
async def test_instantiate_adapters_disabled_skipped(edge_db, tmp_path) -> None:
    """Disabled adapters are skipped during instantiation."""
    from common.config import EdgeConfig

    config_file = tmp_path / "adapters.yaml"
    config_file.write_text("""
adapters:
  - name: disabled_adapter
    enabled: false
    type: generic_modbus
    config:
      host: 192.168.1.1
""")

    config = EdgeConfig(
        device_id="test-edge-001",
        cloud_api_url="https://api.gen318.io",
        device_cert_path="/tmp/fake.pem",
        device_key_path="/tmp/fake.key",
        ca_cert_path="/tmp/fake-ca.pem",
        cert_pin_sha256="",
        sqlite_path="/tmp/test.db",
        sqlite_encryption_key="",
        adapter_config_path=str(config_file),
    )

    service = CollectorService(config)
    service.db = edge_db

    await service._instantiate_adapters()

    assert len(service._managed) == 0


@pytest.mark.asyncio
async def test_instantiate_adapters_unknown_type(edge_db, tmp_path) -> None:
    """Unknown adapter type is logged and skipped."""
    from common.config import EdgeConfig

    config_file = tmp_path / "adapters.yaml"
    config_file.write_text("""
adapters:
  - name: unknown
    enabled: true
    type: nonexistent_adapter
    config: {}
""")

    config = EdgeConfig(
        device_id="test-edge-001",
        cloud_api_url="https://api.gen318.io",
        device_cert_path="/tmp/fake.pem",
        device_key_path="/tmp/fake.key",
        ca_cert_path="/tmp/fake-ca.pem",
        cert_pin_sha256="",
        sqlite_path="/tmp/test.db",
        sqlite_encryption_key="",
        adapter_config_path=str(config_file),
    )

    service = CollectorService(config)
    service.db = edge_db

    await service._instantiate_adapters()

    assert len(service._managed) == 0


def test_adapter_status_in_config_table(edge_db) -> None:
    """Adapter status is persisted to the config table."""
    from common.config import EdgeConfig

    config = EdgeConfig(
        device_id="test-edge-001",
        cloud_api_url="https://api.gen318.io",
        device_cert_path="/tmp/fake.pem",
        device_key_path="/tmp/fake.key",
        ca_cert_path="/tmp/fake-ca.pem",
        cert_pin_sha256="",
        sqlite_path="/tmp/test.db",
        sqlite_encryption_key="",
    )

    service = CollectorService(config)
    service.db = edge_db

    managed = ManagedAdapter(
        name="victron_mqtt",
        adapter_type="victron_mqtt",
        config={},
        status=AdapterStatus.CONNECTED,
        reading_count=42,
    )

    service._update_adapter_status(managed)

    raw = edge_db.get_config("adapter.victron_mqtt.status")
    assert raw is not None
    status = json.loads(raw)
    assert status["name"] == "victron_mqtt"
    assert status["status"] == "connected"
    assert status["reading_count"] == 42


def test_get_adapter_statuses() -> None:
    """get_adapter_statuses returns status for all managed adapters."""
    from common.config import EdgeConfig

    config = EdgeConfig(
        device_id="test-edge-001",
        cloud_api_url="https://api.gen318.io",
        device_cert_path="/tmp/fake.pem",
        device_key_path="/tmp/fake.key",
        ca_cert_path="/tmp/fake-ca.pem",
        cert_pin_sha256="",
        sqlite_path="/tmp/test.db",
        sqlite_encryption_key="",
    )

    service = CollectorService(config)
    service._managed = [
        ManagedAdapter(
            name="a",
            adapter_type="victron_mqtt",
            config={},
            status=AdapterStatus.CONNECTED,
        ),
        ManagedAdapter(
            name="b", adapter_type="byd_can", config={}, status=AdapterStatus.ERROR
        ),
    ]

    statuses = service.get_adapter_statuses()

    assert len(statuses) == 2
    assert statuses[0]["name"] == "a"
    assert statuses[0]["status"] == "connected"
    assert statuses[1]["name"] == "b"
    assert statuses[1]["status"] == "error"
