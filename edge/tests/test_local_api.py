"""Edge local API endpoint tests — all 9 endpoints via ASGI transport."""

from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture
async def api_client(edge_db) -> AsyncGenerator[AsyncClient, None]:
    """Edge API client with seeded test data."""
    now = int(time.time() * 1000)

    # Seed data
    edge_db.insert_measurement(now, "asset-1", "battery.soc_pct", 75.0, 0)
    edge_db.insert_measurement(now, "asset-1", "solar.pv_power_w", 42000.0, 0)
    edge_db.insert_measurement(now - 60000, "asset-1", "battery.soc_pct", 74.0, 0)
    edge_db.insert_event("e1", "inference", "info", {"result": 92.5}, now)
    edge_db.update_model_state("battery_health", "1.0.0", "a" * 64, 95.0)
    edge_db.enqueue_sync(0, {"type": "alert"})
    edge_db.enqueue_sync(2, {"type": "telemetry"})

    mock_mqtt = MagicMock()
    mock_mqtt.connect = MagicMock()
    mock_mqtt.disconnect = MagicMock()
    mock_mqtt.publish = MagicMock()
    mock_mqtt.subscribe = MagicMock()

    with (
        patch.dict(
            os.environ,
            {
                "DEVICE_ID": "test-api-001",
                "SQLITE_PATH": edge_db._db_path,
                "SQLITE_ENCRYPTION_KEY": "",
            },
        ),
        patch("common.mqtt_client.MQTTClient", return_value=mock_mqtt),
    ):
        # Inject globals directly — lifespan won't trigger in ASGI test transport
        import api.main as api_mod

        api_mod._config = __import__(
            "common.config", fromlist=["load_config"]
        ).load_config()
        api_mod._db = edge_db
        api_mod._mqtt = mock_mqtt

        transport = ASGITransport(app=api_mod.app)
        async with AsyncClient(transport=transport, base_url="http://edge") as client:
            yield client

        api_mod._config = None
        api_mod._db = None
        api_mod._mqtt = None


class TestEdgeAPIEndpoints:
    @pytest.mark.asyncio
    async def test_health(self, api_client: AsyncClient) -> None:
        resp = await api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["device_id"] == "test-api-001"
        assert "system" in data
        assert "sync" in data

    @pytest.mark.asyncio
    async def test_metrics_latest(self, api_client: AsyncClient) -> None:
        resp = await api_client.get("/metrics/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 2
        names = {r["metric_name"] for r in data}
        assert "battery.soc_pct" in names

    @pytest.mark.asyncio
    async def test_metrics_history(self, api_client: AsyncClient) -> None:
        resp = await api_client.get(
            "/metrics/history",
            params={"asset_id": "asset-1", "metric": "battery.soc_pct", "hours": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    @pytest.mark.asyncio
    async def test_adapters(self, api_client: AsyncClient) -> None:
        resp = await api_client.get("/adapters")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_models(self, api_client: AsyncClient) -> None:
        resp = await api_client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["model_name"] == "battery_health"

    @pytest.mark.asyncio
    async def test_sync_status(self, api_client: AsyncClient) -> None:
        resp = await api_client.get("/sync/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["queue_depth"]["critical"] >= 1

    @pytest.mark.asyncio
    async def test_events(self, api_client: AsyncClient) -> None:
        resp = await api_client.get("/events", params={"limit": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    @pytest.mark.asyncio
    async def test_events_severity_filter(self, api_client: AsyncClient) -> None:
        resp = await api_client.get("/events", params={"severity": "info", "limit": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert all(e["severity"] == "info" for e in data)

        resp = await api_client.get(
            "/events", params={"severity": "critical", "limit": 10}
        )
        assert resp.status_code == 200
        assert all(e["severity"] == "critical" for e in resp.json())

    @pytest.mark.asyncio
    async def test_diagnostics_run(self, api_client: AsyncClient) -> None:
        resp = await api_client.post("/diagnostics/run")
        assert resp.status_code == 200
        assert resp.json()["status"] == "inference_triggered"

    @pytest.mark.asyncio
    async def test_config(self, api_client: AsyncClient) -> None:
        resp = await api_client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["device_id"] == "test-api-001"
        assert data["sqlite_encryption_key"] == "***"
        assert data["cert_pin_sha256"] == "***"

    @pytest.mark.asyncio
    async def test_energy_and_tuning_on_the_local_page(
        self, api_client: AsyncClient, edge_db, tmp_path, monkeypatch
    ) -> None:
        """The box reports its own energy use and any sensitivity changed by replies."""
        import api.main as api_mod
        from common.energy import EnergyLedger, PowerModel
        from sms.store import SmsStore

        now = time.time()
        ledger = EnergyLedger(tmp_path / "energy.json", PowerModel(measured_watts=6.0))
        for i in range(90):
            ledger.add(0.2, 60, now=now - (90 - i) * 60)
        # one hour of a steady 2 kW site load
        edge_db.insert_measurements([
            (int((now - 3600 + i * 60) * 1000), "asset-1", "load.power_w", 2000.0, 0) for i in range(61)
        ])
        store = SmsStore(str(tmp_path / "sms.db"))
        for i in range(3):
            ref = store.new_ref()
            store.record_alert(ref, "inverter_hot", "warning", "asset-1", {"temp": 71}, now + i)
            store.record_label(ref, 2, "", "+2348030000001", now + i)
        store.close()
        monkeypatch.setattr(api_mod, "ENERGY_LEDGER_PATH", str(tmp_path / "energy.json"))
        monkeypatch.setattr(api_mod, "SMS_DB_PATH", str(tmp_path / "sms.db"))

        data = (await api_client.get("/energy")).json()
        assert data["method"] == "measured" and data["average_watts"] == 6.0
        assert data["wh_per_day"] == 144.0
        assert data["site_wh_last_day"] == pytest.approx(2000, rel=0.02)
        assert data["share_of_site_pct"] == pytest.approx(7.2, rel=0.05)

        tuning = (await api_client.get("/sms/tuning")).json()
        assert tuning == [{"code": "inverter_hot", "steps": 1, "max_steps": 2, "was": "70 C", "now": "73 C"}]

        page = (await api_client.get("/")).text
        assert "6.0 W, 144.0 Wh a day (measured)" in page and "inverter hot" in page

    @pytest.mark.asyncio
    async def test_local_page_before_anything_is_recorded(
        self, api_client: AsyncClient, tmp_path, monkeypatch
    ) -> None:
        import api.main as api_mod

        monkeypatch.setattr(api_mod, "ENERGY_LEDGER_PATH", str(tmp_path / "none.json"))
        monkeypatch.setattr(api_mod, "SMS_DB_PATH", str(tmp_path / "none.db"))
        assert (await api_client.get("/energy")).json()["available"] is False
        assert (await api_client.get("/sms/tuning")).json() == []
        page = (await api_client.get("/")).text
        assert "Not enough samples yet" in page and "Nothing changed" in page
