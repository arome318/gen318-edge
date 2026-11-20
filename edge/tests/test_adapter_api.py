"""Tests for adapter-related edge API endpoints."""

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


TOKEN = "test-token-123"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def token(monkeypatch) -> str:
    monkeypatch.setenv("LOCAL_API_TOKEN", TOKEN)
    return TOKEN


def test_reconnect_not_found(api_client, token) -> None:
    """Returns 404 for unknown adapter."""
    response = api_client.post("/adapters/nonexistent/reconnect", headers=AUTH)
    assert response.status_code == 404


def test_reconnect_success(api_client, edge_db, mock_mqtt, token) -> None:
    """Reconnect publishes MQTT command and returns success."""
    edge_db.set_config(
        "adapter.victron_mqtt.status",
        json.dumps({"name": "victron_mqtt", "status": "error"}),
    )

    response = api_client.post("/adapters/victron_mqtt/reconnect", headers=AUTH)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "reconnect_requested"
    assert data["adapter"] == "victron_mqtt"

    # Verify MQTT publish was called
    mock_mqtt.publish.assert_called()
    call_args = mock_mqtt.publish.call_args
    assert "adapter/reconnect" in call_args[0][0]


# ── Token on the POST endpoints ────────────────────────────────

POSTS = ["/adapters/victron_mqtt/reconnect", "/diagnostics/run"]


@pytest.fixture
def adapter(edge_db) -> None:
    edge_db.set_config(
        "adapter.victron_mqtt.status",
        json.dumps({"name": "victron_mqtt", "status": "error"}),
    )


@pytest.mark.parametrize("path", POSTS)
def test_post_is_switched_off_when_no_token_is_configured(
    api_client, mock_mqtt, adapter, monkeypatch, path
) -> None:
    monkeypatch.delenv("LOCAL_API_TOKEN", raising=False)
    for headers in ({}, AUTH):
        response = api_client.post(path, headers=headers)
        assert response.status_code == 503
        assert "LOCAL_API_TOKEN is not set" in response.json()["detail"]
    monkeypatch.setenv("LOCAL_API_TOKEN", "   ")  # blank counts as not set
    assert api_client.post(path, headers={"X-API-Token": ""}).status_code == 503
    mock_mqtt.publish.assert_not_called()


@pytest.mark.parametrize("path", POSTS)
def test_post_needs_the_right_token(api_client, mock_mqtt, adapter, token, path) -> None:
    missing = api_client.post(path)
    assert missing.status_code == 401 and "Token required" in missing.json()["detail"]
    for headers in ({"Authorization": "Bearer wrong"}, {"X-API-Token": "wrong"},
                    {"Authorization": TOKEN}):  # no "Bearer " prefix
        assert api_client.post(path, headers=headers).status_code == 401
    mock_mqtt.publish.assert_not_called()

    for headers in (AUTH, {"X-API-Token": TOKEN}, {"authorization": f"bearer {TOKEN}"}):
        assert api_client.post(path, headers=headers).status_code == 200
    assert mock_mqtt.publish.call_count == 3


def test_reading_stays_open_without_a_token(api_client, adapter, monkeypatch) -> None:
    """The page and the read-only JSON are for anyone on the site Wi-Fi."""
    monkeypatch.delenv("LOCAL_API_TOKEN", raising=False)
    for path in ("/adapters", "/adapters/victron_mqtt/readings", "/energy", "/sitelearn/status"):
        assert api_client.get(path).status_code == 200


def test_only_the_post_endpoints_change_anything() -> None:
    """Every route that is not a plain read must carry the token guard."""
    import api.main as api_mod

    for route in api_mod.app.routes:
        methods = getattr(route, "methods", set()) - {"GET", "HEAD", "OPTIONS"}
        if methods:
            guards = [d.call for d in route.dependant.dependencies]
            assert api_mod.require_token in guards, route.path
