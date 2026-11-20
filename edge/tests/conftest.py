"""Edge test fixtures — temp SQLite, mocked MQTT/GPIO/Docker."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import pytest

# Ensure edge/ is on the path
EDGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EDGE_DIR not in sys.path:
    sys.path.insert(0, EDGE_DIR)

# Set required env vars before any edge imports
os.environ.setdefault("DEVICE_ID", "test-edge-001")
os.environ.setdefault("CLOUD_API_URL", "https://api.gen318.io")
os.environ.setdefault("DEVICE_CERT_PATH", "/tmp/fake.pem")
os.environ.setdefault("DEVICE_KEY_PATH", "/tmp/fake.key")
os.environ.setdefault("CA_CERT_PATH", "/tmp/fake-ca.pem")
os.environ.setdefault("CERT_PIN_SHA256", "")
os.environ.setdefault("SQLITE_ENCRYPTION_KEY", "")
os.environ.setdefault("MQTT_HOST", "localhost")
os.environ.setdefault("MQTT_PORT", "1883")


@pytest.fixture
def edge_db() -> Generator:
    """Provide a temp SQLite EdgeDatabase with schema initialized."""
    from common.database import EdgeDatabase

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    db = EdgeDatabase(db_path, "")
    db.connect()
    db.init_schema()
    yield db
    db.close()
    os.unlink(db_path)


@pytest.fixture
def edge_config():
    """Provide a test EdgeConfig."""
    from common.config import EdgeConfig

    return EdgeConfig(
        device_id="test-edge-001",
        cloud_api_url="https://api.gen318.io",
        device_cert_path="/tmp/fake.pem",
        device_key_path="/tmp/fake.key",
        ca_cert_path="/tmp/fake-ca.pem",
        cert_pin_sha256="",
        sqlite_path="/tmp/test.db",
        sqlite_encryption_key="",
    )


@pytest.fixture
def mock_mqtt() -> MagicMock:
    """Provide a mocked MQTTClient that captures publishes."""
    mock = MagicMock()
    mock.is_connected = True
    mock.connect = MagicMock()
    mock.disconnect = MagicMock()
    mock.publish = MagicMock()
    mock.subscribe = MagicMock()
    mock._published: list[tuple[str, Any]] = []

    def _capture_publish(topic: str, payload: Any, qos: int = 0) -> None:
        mock._published.append((topic, payload))

    mock.publish.side_effect = _capture_publish
    return mock


@pytest.fixture
def mock_docker_client() -> MagicMock:
    """Provide a mocked Docker client with configurable container states."""
    client = MagicMock()

    def _make_container(name: str, status: str = "running") -> MagicMock:
        container = MagicMock()
        container.name = name
        container.status = status
        container.attrs = {
            "State": {
                "Health": {"Status": "healthy" if status == "running" else "unhealthy"}
            }
        }
        container.restart = MagicMock()
        return container

    containers = {
        "gen318-mqtt": _make_container("gen318-mqtt"),
        "gen318-collector": _make_container("gen318-collector"),
        "gen318-inference": _make_container("gen318-inference"),
        "gen318-sync": _make_container("gen318-sync"),
        "gen318-api": _make_container("gen318-api"),
    }

    def _get(name: str) -> MagicMock:
        if name not in containers:
            import docker.errors

            raise docker.errors.NotFound(f"Container {name} not found")
        return containers[name]

    client.containers.get = _get
    client._containers = containers
    return client
