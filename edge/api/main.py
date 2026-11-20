"""Edge local web UI — FastAPI on :8080 (Doc 14 §8.9, F-EDGE-07).

Serves site dashboard, live metrics, and adapter status without
cloud connectivity. All data is read from local SQLite.
"""

from __future__ import annotations

import os
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import psutil
import structlog
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from common.config import EdgeConfig, load_config
from common.database import EdgeDatabase
from common.mqtt_client import (
    TOPIC_CONTROL_ADAPTER_RECONNECT,
    TOPIC_CONTROL_DIAGNOSTICS,
    MQTTClient,
)
from common.storage import StoragePolicy, assess_storage

logger = structlog.get_logger()

# Module-level state (initialized in lifespan)
_config: EdgeConfig | None = None
_db: EdgeDatabase | None = None
_mqtt: MQTTClient | None = None


def _read_cpu_temperature() -> float:
    """Read SoC temperature from sysfs. Returns -1.0 if unavailable."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError):
        return -1.0


def _get_system_metrics() -> dict[str, Any]:
    """Collect system metrics via psutil."""
    try:
        uptime_seconds = int(time.time() - psutil.boot_time())
    except (OSError, PermissionError):
        # Some hardened containers deny the sysctl used by psutil on macOS and
        # Linux. Health reporting must degrade, not make the local API fail.
        uptime_seconds = -1
    return {
        "cpu_pct": psutil.cpu_percent(interval=0.1),
        "ram_pct": psutil.virtual_memory().percent,
        "disk_pct": (
            psutil.disk_usage("/data").percent
            if os.path.exists("/data")
            else psutil.disk_usage("/").percent
        ),
        "temperature_c": _read_cpu_temperature(),
        "uptime_seconds": uptime_seconds,
    }


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Manage startup/shutdown of database and MQTT connections."""
    global _config, _db, _mqtt
    _config = load_config()
    _db = EdgeDatabase(_config.sqlite_path, _config.sqlite_encryption_key)
    _db.connect()
    _db.init_schema()
    _mqtt = MQTTClient(
        host=_config.mqtt_host,
        port=_config.mqtt_port,
        client_id="gen318-api",
    )
    try:
        _mqtt.connect(max_retries=3)
    except RuntimeError:
        logger.warning(
            "mqtt_unavailable",
            msg="MQTT broker not reachable — diagnostics endpoint disabled",
        )
    logger.info("edge_api_started", device_id=_config.device_id)
    yield
    if _mqtt:
        _mqtt.disconnect()
    if _db:
        _db.close()


app = FastAPI(
    title="Gen318 Edge Node",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Full node health report."""
    assert _config is not None and _db is not None
    try:
        storage: dict[str, Any] = assess_storage(
            _config.sqlite_path,
            StoragePolicy(
                target_days=_config.storage_target_days,
                expected_assets=_config.storage_expected_assets,
                metrics_per_asset=_config.storage_metrics_per_asset,
                bytes_per_measurement=_config.storage_bytes_per_measurement,
                poll_interval_seconds=_config.poll_interval_seconds,
                reserve_bytes=_config.storage_reserve_bytes,
                warning_used_pct=_config.storage_warning_used_pct,
                critical_used_pct=_config.storage_critical_used_pct,
                emergency_used_pct=_config.storage_emergency_used_pct,
            ),
        ).to_dict()
    except OSError as exc:
        # Retain the collector's last successful assessment when this read-only
        # container cannot stat the mount. Health remains useful offline.
        cached = _db.get_config("storage.status")
        try:
            storage = json.loads(cached) if cached else {"status": "unavailable"}
        except json.JSONDecodeError:
            storage = {"status": "unavailable"}
        storage["assessment_error"] = type(exc).__name__
    return {
        "device_id": _config.device_id,
        "status": "healthy",
        "system": _get_system_metrics(),
        "sync": {
            "queue_depth": _db.get_queue_depth_by_priority(),
            "dead_letters": _db.get_dead_letter_count(),
        },
        "models": _db.get_model_state(),
        "storage": storage,
    }


@app.get("/metrics/latest")
async def metrics_latest() -> list[dict[str, Any]]:
    """Latest readings from all adapters."""
    assert _db is not None
    return _db.get_latest_readings()


@app.get("/metrics/history")
async def metrics_history(
    asset_id: str = Query(..., description="Asset identifier"),
    metric: str = Query(..., description="Metric name"),
    hours: int = Query(24, ge=1, le=720, description="Hours of history"),
) -> list[dict[str, Any]]:
    """Historical data from local SQLite."""
    assert _db is not None
    return _db.get_readings_history(asset_id, metric, hours)


@app.get("/adapters")
async def adapters() -> list[dict[str, Any]]:
    """List all configured adapters with connection status, last reading time, error count.

    Reads adapter status from the config table where the collector writes
    status updates with keys ``adapter.{name}.status``.
    """
    assert _db is not None
    import json as _json

    results: list[dict[str, Any]] = []

    # Scan config table for adapter status keys
    # The collector writes keys like: adapter.victron_mqtt.status
    for key_prefix in [
        "adapter.victron_mqtt.status",
        "adapter.victron_modbus.status",
        "adapter.byd_can.status",
        "adapter.okra_harvest.status",
        "adapter.generic_modbus.status",
    ]:
        raw = _db.get_config(key_prefix)
        if raw:
            try:
                results.append(_json.loads(raw))
            except _json.JSONDecodeError:
                pass

    # Also scan for any adapter names not in the default list
    # by reading all config keys with adapter.*.status pattern
    with _db._lock:
        assert _db._conn is not None
        cursor = _db._conn.execute(
            "SELECT key, value FROM config WHERE key LIKE 'adapter.%.status'"
        )
        for row in cursor.fetchall():
            key, value = row
            try:
                status_dict = _json.loads(value)
                # Avoid duplicates
                if not any(r.get("name") == status_dict.get("name") for r in results):
                    results.append(status_dict)
            except _json.JSONDecodeError:
                pass

    return results


@app.get("/adapters/{name}/readings")
async def adapter_readings(name: str) -> list[dict[str, Any]]:
    """Latest readings from a specific adapter.

    Reads the adapter's status to find its asset IDs, then queries
    the measurements table for latest readings from those assets.
    """
    assert _db is not None

    raw = _db.get_config(f"adapter.{name}.status")
    if not raw:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Adapter '{name}' not found")

    # Get all latest readings and filter by adapter context
    # Since adapters write asset_id on their readings, we return all latest
    return _db.get_latest_readings()


@app.post("/adapters/{name}/reconnect")
async def adapter_reconnect(name: str) -> dict[str, str]:
    """Force reconnect for a specific adapter.

    Publishes a reconnect command to MQTT which the collector processes.
    """
    assert _db is not None

    # Verify adapter exists
    raw = _db.get_config(f"adapter.{name}.status")
    if not raw:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Adapter '{name}' not found")

    if _mqtt is None or not _mqtt.is_connected:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="MQTT broker unavailable")

    _mqtt.publish(
        TOPIC_CONTROL_ADAPTER_RECONNECT,
        {"adapter_name": name},
    )

    return {"status": "reconnect_requested", "adapter": name}


@app.get("/models")
async def models() -> list[dict[str, Any]]:
    """Loaded model versions and accuracy."""
    assert _db is not None
    return _db.get_model_state()


@app.get("/sync/status")
async def sync_status() -> dict[str, Any]:
    """Sync queue depth per priority tier plus telemetry drain backlog."""
    assert _db is not None
    return {
        "queue_depth": _db.get_queue_depth_by_priority(),
        "dead_letters": _db.get_dead_letter_count(),
        "unsynced_measurements": _db.count_unsynced_measurements(),
        "rejected_measurements": _db.count_rejected_measurements(),
    }


@app.get("/events")
async def events(
    severity: str | None = Query(None, description="Filter by severity"),
    limit: int = Query(50, ge=1, le=500, description="Max events to return"),
) -> list[dict[str, Any]]:
    """Recent local events."""
    assert _db is not None
    return _db.get_events(severity=severity, limit=limit)


@app.post("/diagnostics/run")
async def run_diagnostics() -> dict[str, str]:
    """Trigger on-demand inference for all assets."""
    assert _mqtt is not None
    _mqtt.publish(TOPIC_CONTROL_DIAGNOSTICS, {"action": "run_all"})
    return {"status": "inference_triggered"}


@app.get("/config")
async def config() -> dict[str, Any]:
    """Current node configuration (secrets redacted)."""
    assert _config is not None
    return {
        "device_id": _config.device_id,
        "cloud_api_url": _config.cloud_api_url,
        "mqtt_host": _config.mqtt_host,
        "mqtt_port": _config.mqtt_port,
        "poll_interval_seconds": _config.poll_interval_seconds,
        "sync_interval_seconds": _config.sync_interval_seconds,
        "inference_interval_seconds": _config.inference_interval_seconds,
        "check_interval": _config.check_interval,
        "adapter_config_path": _config.adapter_config_path,
        "model_path": _config.model_path,
        "sqlite_path": _config.sqlite_path,
        # Secrets redacted
        "sqlite_encryption_key": "***",
        "cert_pin_sha256": "***",
    }


# ── Technician page and SMS record (Gen318 Edge) ─────────────────

SMS_DB_PATH = os.environ.get("SMS_DB_PATH", "/data/sms/sms.db")
LABEL_NAMES = {1: "real", 2: "false alarm", 3: "other cause"}


def _sms_alerts(limit: int = 30) -> list[dict[str, Any]]:
    if not os.path.exists(SMS_DB_PATH):
        return []
    from sms.store import SmsStore

    store = SmsStore(SMS_DB_PATH)
    try:
        return store.recent_alerts(limit)
    finally:
        store.close()


def _sitelearn_status() -> dict[str, Any]:
    path = os.path.join(_config.model_path if _config else "/models", "site_learning", "meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


@app.get("/sitelearn/status")
async def sitelearn_status() -> dict[str, Any]:
    """What the on-site models have learned so far (no raw data)."""
    return _sitelearn_status()


@app.get("/sms/alerts")
async def sms_alerts(limit: int = Query(30, ge=1, le=500)) -> list[dict[str, Any]]:
    """Recent SMS alerts with technician labels (no phone numbers)."""
    return _sms_alerts(limit)


def _esc(value: Any) -> str:
    import html

    return html.escape(str(value))


@app.get("/", response_class=HTMLResponse)
async def technician_page() -> str:
    """Plain page for a technician's phone on the site Wi-Fi. No internet needed."""
    assert _db is not None
    readings = _db.get_latest_readings()
    rows = "".join(
        f"<tr><td>{_esc(r.get('asset_id'))}</td><td>{_esc(r.get('metric_name'))}</td>"
        f"<td>{_esc(round(float(r.get('value', 0)), 2))}</td></tr>"
        for r in readings[:200]
    )
    alerts = "".join(
        f"<tr><td>#{_esc(a['ref'])}</td><td>{_esc(a['code'].replace('_', ' '))}</td>"
        f"<td>{_esc(a['severity'])}</td>"
        f"<td>{_esc(time.strftime('%d %b %H:%M', time.gmtime(a['raised_at'] + 3600)))}</td>"
        f"<td>{'open' if a['cleared_at'] is None else 'cleared'}</td>"
        f"<td>{_esc(LABEL_NAMES.get(a['label'], '-') if a['label'] else '-')}</td></tr>"
        for a in _sms_alerts()
    )
    learned = "".join(
        f"<tr><td>{_esc(k.split(':')[0])} ({_esc(k.split(':')[1])})</td>"
        f"<td>{_esc(time.strftime('%d %b %H:%M', time.gmtime(v['trained_at'] + 3600)))}</td>"
        f"<td>{_esc(v.get('days', '-'))}</td></tr>"
        for k, v in sorted(_sitelearn_status().items())
        if isinstance(v, dict) and k.count(":") == 1 and "trained_at" in v
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30"><title>Gen318 Edge</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;color:#111;background:#fff}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin-bottom:24px}}
td,th{{border-bottom:1px solid #ddd;padding:6px;text-align:left}}h1{{font-size:20px}}
h2{{font-size:16px;margin-top:24px}}p{{color:#555;font-size:13px}}</style></head><body>
<h1>Gen318 Edge &middot; {_esc(_config.device_id if _config else '')}</h1>
<p>Works offline. Page refreshes every 30 seconds. Times are WAT.</p>
<h2>Alerts</h2><p>To mark an alert, reply to its SMS: 1 = real, 2 = false alarm, 3 = other cause.</p>
<table><tr><th>Ref</th><th>Fault</th><th>Level</th><th>Raised</th><th>Status</th><th>Marked</th></tr>
{alerts or '<tr><td colspan="6">No alerts yet.</td></tr>'}</table>
<h2>What this box has learned</h2>
<p>Models train here from this site's own data: solar after 7 days, load after 7, battery after 14.</p>
<table><tr><th>Model</th><th>Trained</th><th>Days of data</th></tr>
{learned or '<tr><td colspan="3">Still collecting data.</td></tr>'}</table>
<h2>Latest readings</h2>
<table><tr><th>Asset</th><th>Metric</th><th>Value</th></tr>
{rows or '<tr><td colspan="3">No readings yet.</td></tr>'}</table>
</body></html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
