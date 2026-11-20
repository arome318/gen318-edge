"""Tests for adapter-related edge API endpoints (Doc 15 §5)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(edge_db, mock_mqtt):
    """Create a test client with mocked DB and MQTT."""
    import api.main as api_module

    api_module._db = edge_db
    api_module._mqtt = mock_mqtt
    api_module._config = MagicMock()
    api_module._config.device_id = "test-edge-001"

    client = TestClient(api_module.app)
    yield client

    # Reset module state
    api_module._db = None
    api_module._mqtt = None
    api_module._config = None


# ── GET /adapters Tests ────────────────────────────────────────


def test_get_adapters_empty(api_client, edge_db) -> None:
    """Returns empty list when no adapter statuses in config."""
    response = api_client.get("/adapters")
    assert response.status_code == 200
    assert response.json() == []


def test_get_adapters_with_status(api_client, edge_db) -> None:
    """Returns adapter statuses from config table."""
    status = {
        "name": "victron_mqtt",
        "type": "victron_mqtt",
        "status": "connected",
        "error_count": 0,
        "last_reading_at": 1710000000.0,
        "reading_count": 42,
    }
    edge_db.set_config(
        "adapter.victron_mqtt.status",
        json.dumps(status),
    )

    response = api_client.get("/adapters")
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "victron_mqtt"
    assert data[0]["status"] == "connected"
    assert data[0]["reading_count"] == 42


def test_get_adapters_multiple(api_client, edge_db) -> None:
    """Returns statuses for multiple adapters."""
    for name, status in [
        ("victron_mqtt", "connected"),
        ("byd_can", "error"),
    ]:
        edge_db.set_config(
            f"adapter.{name}.status",
            json.dumps(
                {
                    "name": name,
                    "type": name,
                    "status": status,
                    "error_count": 0 if status == "connected" else 5,
                    "last_reading_at": 1710000000.0,
                    "reading_count": 100,
                }
            ),
        )

    response = api_client.get("/adapters")
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 2
    names = {d["name"] for d in data}
    assert "victron_mqtt" in names
    assert "byd_can" in names


# ── GET /adapters/{name}/readings Tests ────────────────────────


def test_get_adapter_readings_not_found(api_client) -> None:
    """Returns 404 for unknown adapter."""
    response = api_client.get("/adapters/nonexistent/readings")
    assert response.status_code == 404


def test_get_adapter_readings_exists(api_client, edge_db) -> None:
    """Returns readings when adapter exists."""
    edge_db.set_config(
        "adapter.victron_mqtt.status",
        json.dumps({"name": "victron_mqtt", "status": "connected"}),
    )

    # Insert some test readings
    import time

    now_ms = int(time.time() * 1000)
    edge_db.insert_measurements(
        [
            (now_ms, "asset-001", "battery.soc_pct", 85.0, 0),
            (now_ms, "asset-002", "solar.pv_power_w", 3500.0, 0),
        ]
    )

    response = api_client.get("/adapters/victron_mqtt/readings")
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 2


# ── POST /adapters/{name}/reconnect Tests ──────────────────────


def test_reconnect_not_found(api_client) -> None:
    """Returns 404 for unknown adapter."""
    response = api_client.post("/adapters/nonexistent/reconnect")
    assert response.status_code == 404


def test_reconnect_success(api_client, edge_db, mock_mqtt) -> None:
    """Reconnect publishes MQTT command and returns success."""
    edge_db.set_config(
        "adapter.victron_mqtt.status",
        json.dumps({"name": "victron_mqtt", "status": "error"}),
    )

    response = api_client.post("/adapters/victron_mqtt/reconnect")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "reconnect_requested"
    assert data["adapter"] == "victron_mqtt"

    # Verify MQTT publish was called
    mock_mqtt.publish.assert_called()
    call_args = mock_mqtt.publish.call_args
    assert "adapter/reconnect" in call_args[0][0]
