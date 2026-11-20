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
    async def test_diagnostics_run(self, api_client: AsyncClient, monkeypatch) -> None:
        monkeypatch.setenv("LOCAL_API_TOKEN", "t0ken")
        assert (await api_client.post("/diagnostics/run")).status_code == 401
        resp = await api_client.post("/diagnostics/run", headers={"X-API-Token": "t0ken"})
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

        store = SmsStore(str(tmp_path / "sms.db"))
        store.record_delivery(None, "+2348030000001", "technician", "alert", True, now, "Kuje: Power is OFF <b>")
        store.record_inbound("+2348030000009", "status", now + 1)
        store.close()
        msgs = (await api_client.get("/sms/messages")).json()
        assert [m["direction"] for m in msgs[:2]] == ["in", "out"]
        page = (await api_client.get("/")).text
        assert "Kuje: Power is OFF &lt;b&gt;" in page and "Someone asked for power status" in page

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
        assert "Not enough samples yet" in page and "Nothing changed" in page and "No SMS yet" in page


class TestLocalPage:
    """The technician page: renders, is self-contained, and tells stale data from live."""

    @pytest.mark.asyncio
    async def test_page_renders_with_brand_and_live_figures(self, api_client: AsyncClient) -> None:
        resp = await api_client.get("/")
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/html")
        page = resp.text
        assert 'aria-label="Gen318"' in page and 'class="lockup"' in page      # logo inline, no image request
        assert 'href="static/page.css"' in page and 'src="static/page.js"' in page
        assert 'data-refresh="30"' in page and "<noscript>" in page
        # KPI tiles from the seeded readings: battery 75 %, solar 42 kW
        assert '<div class="card kpi battery">' in page and "75<small>%</small>" in page
        assert "42.00<small>kW</small>" in page
        assert 'class="chip ok"' in page and "Comms OK" in page and 'class="kpis"' in page
        for block in ("kpis", "comms", "feed", "alerts", "learned", "tuning", "energy", "readings"):
            assert f'id="{block}" data-live' in page
        # the raw readings table is there but folded away, so the SMS feed keeps the eye
        assert "<details><summary><h2>Latest readings</h2></summary>" in page and "<details open" not in page
        assert "<td>asset-1</td>" in page          # asset ids are printed as stored

    @pytest.mark.asyncio
    async def test_page_needs_nothing_from_the_internet(self, api_client: AsyncClient) -> None:
        """A box on a mini-grid has no internet: no CDN, no web fonts, no remote script."""
        import re
        from pathlib import Path

        import api.main as api_mod

        static = Path(api_mod.__file__).parent / "static"
        remote_url = re.compile(r"""(?:https?:)?//[\w.-]+\.[a-z]{2,}[^\s"')]*""")
        assert remote_url.findall('<link href="https://fonts.googleapis.com/css2?family=X">')  # the check can fail
        sources = [(await api_client.get("/")).text, (static / "page.css").read_text(),
                   (static / "page.js").read_text()]
        for text in sources:
            remote = [u for u in remote_url.findall(text) if "www.w3.org" not in u]
            assert remote == []
        for path, kind in (("/static/page.css", "text/css"), ("/static/page.js", "javascript"),
                           ("/static/favicon.svg", "svg"),
                           ("/static/fonts/ibm-plex-mono-latin-600-normal.woff2", "font"),
                           ("/static/fonts/archivo-latin-wdth-normal.woff2", "font")):
            resp = await api_client.get(path)
            assert resp.status_code == 200 and kind in resp.headers["content-type"], path
        # every font the stylesheet names is bundled, with its licence
        css = (static / "page.css").read_text()
        assert ".readings td:first-child::first-letter { text-transform: none; }" in css   # no "Inv-1"
        fonts = re.findall(r'url\("fonts/([^"]+)"\)', css)
        assert len(fonts) == 7
        for name in fonts:
            assert (static / "fonts" / name).is_file(), name
        assert (static / "fonts" / "OFL-IBM-Plex.txt").is_file() and (static / "fonts" / "OFL-Archivo.txt").is_file()

    @pytest.mark.asyncio
    async def test_fragment_and_theme(self, api_client: AsyncClient) -> None:
        full = (await api_client.get("/")).text
        frag = (await api_client.get("/", params={"fragment": 1})).text
        assert "<script" not in frag and "<head>" not in frag and 'id="feed" data-live' in frag
        assert frag.count("data-live") == full.count("data-live")
        assert "data-theme" not in full
        assert '<html lang="en" data-theme="dark"' in (await api_client.get("/", params={"theme": "dark"})).text
        assert 'data-theme="light"' in (await api_client.get("/", params={"theme": "light"})).text
        assert (await api_client.get("/", params={"theme": "pink"})).status_code == 422

    @pytest.mark.asyncio
    async def test_stale_readings_dim_and_lost_comms_shows(
        self, api_client: AsyncClient, tmp_path, monkeypatch
    ) -> None:
        import api.main as api_mod

        # fresh readings, 30-second comms-lost threshold: fine
        cfg = tmp_path / "sms.yaml"
        cfg.write_text("site_name: Kuje <Mini-grid>\nthresholds: {comms_lost_s: 30}\n")
        monkeypatch.setenv("SMS_CONFIG_PATH", str(cfg))
        page = (await api_client.get("/")).text
        assert "<h1>Kuje &lt;Mini-grid&gt;</h1>" in page and "Comms OK" in page

        real = api_mod.time.time
        monkeypatch.setattr(api_mod.time, "time", lambda: real() + 120)   # two minutes with no reading
        page = (await api_client.get("/")).text
        assert 'class="chip danger"' in page and "Comms lost &middot; 2 min" in page
        assert 'class="kpis stale"' in page and "As of " in page

    @pytest.mark.asyncio
    async def test_sms_feed_is_bubbles_with_who_and_when(
        self, api_client: AsyncClient, tmp_path, monkeypatch
    ) -> None:
        import api.main as api_mod
        from sms.store import SmsStore

        now = time.time()
        store = SmsStore(str(tmp_path / "sms.db"))
        ref = store.new_ref()
        store.record_alert(ref, "comms_lost", "critical", "site", {"minutes": 1}, now)
        store.record_delivery(ref, "+2348030000001", "technician", "alert", True, now, "Kuje: Box cannot read equipment")
        store.record_inbound("+2348030000009", "status", now + 1)
        store.record_delivery(None, "+2348030000009", "community", "reply", True, now + 1, "Kuje: Power ON.")
        store.record_inbound("+2348030000001", "label", now + 2)
        store.record_label(ref, 1, "", "+2348030000001", now + 2)
        store.close()
        monkeypatch.setattr(api_mod, "SMS_DB_PATH", str(tmp_path / "sms.db"))

        page = (await api_client.get("/")).text
        feed = page[page.index('id="feed"'):page.index('class="foot"', page.index('id="feed"'))]
        assert feed.count('class="msg out"') == 2 and feed.count('class="msg in"') == 2
        assert "Box &rarr; technician" in feed and "Box &rarr; community" in feed
        assert '<div class="bubble">POWER</div>' in feed and "Phone &rarr; box" in feed
        assert '<div class="bubble">1</div>' in feed and "Technician &rarr; box" in feed
        # oldest first: the alert, the question, its reply, then the technician's 1
        order = [feed.index(x) for x in ("cannot read equipment", ">POWER<", "Power ON.", ">1<")]
        assert order == sorted(order)
        assert "2348030000001" not in page and "2348030000009" not in page
        assert '<span class="badge critical">critical</span>' in page and "marked real" in page
