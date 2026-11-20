"""Tests for the sync service's MQTT→queue bridging.

The watchdog publishes tamper/critical alerts on gen318/alerts/critical
(QoS 2); the sync service must land them in the P0 sync queue so they
drain to POST /ingest/alert — locally durable, offline-safe.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from sync.engine import Priority


def _make_service(edge_config, edge_db):
    """Build a SyncService without touching MQTT/HTTP."""
    from sync.main import SyncService

    svc = SyncService.__new__(SyncService)
    svc.config = edge_config
    svc.db = edge_db
    svc.mqtt = MagicMock()
    svc.engine = MagicMock()
    svc._loop = None
    return svc


def test_critical_alert_lands_in_p0_queue(edge_config, edge_db):
    svc = _make_service(edge_config, edge_db)
    payload = {
        "type": "tamper_detected",
        "severity": "critical",
        "device_id": "test-edge-001",
        "gpio_pin": 17,
    }

    svc._on_critical_alert(payload)

    items = edge_db.get_queue_items(priority=Priority.CRITICAL)
    assert len(items) == 1
    assert items[0]["payload"] == payload
    # Other tiers untouched
    assert edge_db.get_queue_items(priority=Priority.HIGH) == []


def test_critical_alert_without_loop_still_enqueues(edge_config, edge_db):
    """The bridge must work before run() sets the loop — data first."""
    svc = _make_service(edge_config, edge_db)
    svc._loop = None

    svc._on_critical_alert({"type": "watchdog_alert", "severity": "critical"})

    assert len(edge_db.get_queue_items(priority=Priority.CRITICAL)) == 1


@pytest.mark.asyncio
async def test_critical_alert_requests_immediate_flush(edge_config, edge_db):
    svc = _make_service(edge_config, edge_db)
    svc._loop = asyncio.get_running_loop()

    svc._on_critical_alert({"type": "tamper_detected", "severity": "critical"})

    # call_soon_threadsafe schedules onto the loop — give it one tick
    await asyncio.sleep(0)
    svc.engine.request_flush.assert_called_once()


def test_enqueue_failure_does_not_raise(edge_config, edge_db):
    """MQTT callbacks must never propagate exceptions into paho's thread."""
    svc = _make_service(edge_config, edge_db)
    svc.db = MagicMock()
    svc.db.enqueue_sync.side_effect = RuntimeError("disk full")

    svc._on_critical_alert({"type": "tamper_detected"})  # must not raise

    svc.engine.request_flush.assert_not_called()


@pytest.mark.asyncio
async def test_run_subscribes_to_critical_alerts_qos2(edge_config, edge_db):
    """run() must wire the alerts subscription at QoS 2 (matches publisher)."""
    from unittest.mock import AsyncMock

    from common.mqtt_client import TOPIC_ALERTS_CRITICAL

    svc = _make_service(edge_config, edge_db)
    svc.db = MagicMock()
    svc.engine = MagicMock()
    svc.engine.sync_loop = AsyncMock()
    svc.engine.close = AsyncMock()

    await svc.run()

    alert_calls = [
        c
        for c in svc.mqtt.subscribe.call_args_list
        if c.args[0] == TOPIC_ALERTS_CRITICAL
    ]
    assert len(alert_calls) == 1
    assert alert_calls[0].args[1] == svc._on_critical_alert
    assert alert_calls[0].kwargs.get("qos") == 2


@pytest.mark.asyncio
async def test_request_flush_wakes_sync_loop_early(edge_db):
    """request_flush() must cut the inter-cycle sleep short."""
    from sync.engine import SyncConfig, SyncEngine

    config = SyncConfig(
        cloud_api_url="https://api.gen318.io",
        device_id="test-edge-001",
        device_cert_path="/tmp/fake.pem",
        device_key_path="/tmp/fake.key",
        ca_cert_path="/tmp/fake-ca.pem",
        cert_pin_sha256="",
        sync_interval=300,
    )
    engine = SyncEngine(config, edge_db)
    cycles: list[int] = []

    async def fake_cycle(cycle: int, mqtt_client=None) -> None:
        cycles.append(cycle)

    engine._sync_cycle = fake_cycle  # type: ignore[method-assign]

    task = asyncio.create_task(engine.sync_loop())
    try:
        # First cycle runs immediately, then the loop sleeps 300s
        await asyncio.sleep(0.05)
        assert cycles == [1]

        engine.request_flush()
        await asyncio.sleep(0.05)
        assert cycles == [1, 2], "wakeup did not interrupt the sleep"
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await engine.close()
