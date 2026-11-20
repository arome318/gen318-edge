"""Edge local web UI — FastAPI on :8080.

Serves site dashboard, live metrics, and adapter status without
cloud connectivity. All data is read from local SQLite.
"""

from __future__ import annotations

import hmac
import os
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import psutil
import structlog
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query
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
    if not os.environ.get(API_TOKEN_ENV, "").strip():
        logger.warning("api_token_not_set",
                       msg=f"{API_TOKEN_ENV} is not set: POST endpoints are switched off")
    logger.info("edge_api_started", device_id=_config.device_id)
    yield
    if _mqtt:
        _mqtt.disconnect()
    if _db:
        _db.close()


API_TOKEN_ENV = "LOCAL_API_TOKEN"


def require_token(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None),
) -> None:
    """Guard for every endpoint that changes something (the POST endpoints).

    Reading the page and the JSON stays open to anyone on the site Wi-Fi.
    Anything that makes the box act needs the token set in LOCAL_API_TOKEN,
    sent as ``Authorization: Bearer <token>`` or ``X-API-Token: <token>``.
    With no token configured these endpoints are switched off, not left open.
    """
    expected = os.environ.get(API_TOKEN_ENV, "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=(f"This action is switched off: {API_TOKEN_ENV} is not set on the box. "
                    f"Set {API_TOKEN_ENV} in the box's .env file and restart the api container."),
        )
    supplied = (x_api_token or "").strip()
    if not supplied and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied:
        raise HTTPException(
            status_code=401,
            detail="Token required: send 'Authorization: Bearer <token>' or 'X-API-Token: <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Wrong token.",
                            headers={"WWW-Authenticate": "Bearer"})


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
        raise HTTPException(status_code=404, detail=f"Adapter '{name}' not found")

    # Get all latest readings and filter by adapter context
    # Since adapters write asset_id on their readings, we return all latest
    return _db.get_latest_readings()


@app.post("/adapters/{name}/reconnect", dependencies=[Depends(require_token)])
async def adapter_reconnect(name: str) -> dict[str, str]:
    """Force reconnect for a specific adapter.

    Publishes a reconnect command to MQTT which the collector processes.
    """
    assert _db is not None

    # Verify adapter exists
    raw = _db.get_config(f"adapter.{name}.status")
    if not raw:
        raise HTTPException(status_code=404, detail=f"Adapter '{name}' not found")

    if _mqtt is None or not _mqtt.is_connected:
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


@app.post("/diagnostics/run", dependencies=[Depends(require_token)])
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
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@app.get("/sitelearn/status")
async def sitelearn_status() -> dict[str, Any]:
    """What the on-site models have learned so far (no raw data)."""
    return _sitelearn_status()


INBOUND_NAMES = {"status": "asked for power status", "label": "marked an alert",
                 "lang": "changed language", "reset": "reset alert sensitivity",
                 "help": "asked for help", "unknown": "sent a message the box did not understand"}
PAGE_REFRESH_SECONDS = max(2, int(os.environ.get("PAGE_REFRESH_SECONDS", "30")))


def _sms_messages(limit: int = 20) -> list[dict[str, Any]]:
    if not os.path.exists(SMS_DB_PATH):
        return []
    from sms.store import SmsStore

    store = SmsStore(SMS_DB_PATH)
    try:
        return store.recent_messages(limit)
    finally:
        store.close()


@app.get("/sms/messages")
async def sms_messages(limit: int = Query(20, ge=1, le=200)) -> list[dict[str, Any]]:
    """Recent SMS: text of what the box sent, kind of what it received (no phone numbers)."""
    return _sms_messages(limit)


def _sms_tuning() -> list[dict[str, Any]]:
    """Warnings made less sensitive by technician replies (see sms/tuning.py)."""
    cfg_path = os.environ.get("SMS_CONFIG_PATH", "/config/sms.yaml")
    if not os.path.exists(SMS_DB_PATH):
        return []
    from sms.rules import Thresholds
    from sms.store import SmsStore
    from sms.tuning import decide, describe

    base = Thresholds()
    try:
        import yaml

        with open(cfg_path) as fh:
            raw = yaml.safe_load(fh) or {}
        if raw.get("auto_tune", True) is False:
            return []
        base = Thresholds.from_dict(raw.get("thresholds"))
    except (OSError, ValueError):
        pass  # page still shows the steps, against default thresholds
    store = SmsStore(SMS_DB_PATH)
    try:
        since = float(store.get_meta("tuning_reset_at") or 0.0)
        return describe(base, decide(store.labels_by_code(since)))
    finally:
        store.close()


ENERGY_LEDGER_PATH = os.environ.get("ENERGY_LEDGER_PATH", "/data/metrics/energy.json")
_SITE_LOAD_METRICS = ("system.ac_consumption_w", "load.power_w", "inverter.ac_output_power_w")


def _site_wh_last_day() -> float | None:
    """Energy the site consumed in the last 24 hours, from the stored load readings."""
    if _db is None:
        return None
    total, found = 0.0, False
    by_asset: dict[str, set[str]] = {}
    for r in _db.get_latest_readings():
        by_asset.setdefault(r["asset_id"], set()).add(r["metric_name"])
    for asset, metrics in by_asset.items():
        metric = next((m for m in _SITE_LOAD_METRICS if m in metrics), None)
        if metric is None:
            continue
        ts, vals = _db.get_series(asset, metric, 24)
        if len(ts) < 2:
            continue
        gaps = [min(float(b - a), 900.0) for a, b in zip(ts[:-1], ts[1:], strict=True)]
        total += sum(float(v) * g for v, g in zip(vals[:-1], gaps, strict=True)) / 3600
        found = True
    return total if found else None


def _energy() -> dict[str, Any]:
    from common.energy import read_ledger, summarise

    out = summarise(read_ledger(ENERGY_LEDGER_PATH), _site_wh_last_day())
    learned = [v for v in _sitelearn_status().values() if isinstance(v, dict)]
    out["models"] = {
        "count": sum(1 for v in learned if "bytes" in v),
        "bytes": sum(int(v.get("bytes", 0)) for v in learned),
        "last_training_seconds": round(sum(float(v.get("fit_s", 0.0)) for v in learned), 2),
    }
    return out


@app.get("/energy")
async def energy() -> dict[str, Any]:
    """What the box itself uses: watts, Wh per day, share of site energy, model size."""
    return _energy()


@app.get("/sms/tuning")
async def sms_tuning() -> list[dict[str, Any]]:
    """Warnings whose sensitivity technician replies have changed."""
    return _sms_tuning()


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
    def _when(ts: float) -> str:
        return time.strftime("%d %b %H:%M:%S", time.gmtime(ts + 3600))

    messages = "".join(
        (f"<tr><td>{_esc(_when(m['at']))}</td><td>Box &rarr; {_esc(m['role'])}</td>"
         f"<td>{_esc(m['text'] or '(' + m['kind'] + ')')}{'' if m['ok'] else ' <b>(not sent)</b>'}</td></tr>")
        if m["direction"] == "out" else
        (f"<tr class=\"in\"><td>{_esc(_when(m['at']))}</td><td>Phone &rarr; box</td>"
         f"<td><i>Someone {_esc(INBOUND_NAMES.get(m['kind'], m['kind']))}</i></td></tr>")
        for m in _sms_messages()
    )
    tuning = "".join(
        f"<tr><td>{_esc(t['code'].replace('_', ' '))}</td><td>{_esc(t['steps'])} of {_esc(t['max_steps'])}</td>"
        f"<td>{_esc(t.get('was', '-'))}</td><td>{_esc(t.get('now', 'set by the model'))}</td></tr>"
        for t in _sms_tuning()
    )
    en = _energy()
    if en.get("available"):
        share = (f", about {_esc(en['share_of_site_pct'])}% of what the site used in the last day"
                 if "share_of_site_pct" in en else "")
        box_energy = (f"About {_esc(en['average_watts'])} W, {_esc(en['wh_per_day'])} Wh a day "
                      f"({_esc(en['method'])}){share}.")
    else:
        box_energy = "Not enough samples yet (needs about an hour)."
    models = en["models"]
    box_models = (f"{_esc(models['count'])} model files, {_esc(round(models['bytes'] / 1024))} kB in all; "
                  f"the last training took {_esc(models['last_training_seconds'])} s on this box."
                  if models["count"] else "No models trained yet.")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{PAGE_REFRESH_SECONDS}"><title>Gen318 Edge</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;color:#111;background:#fff}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin-bottom:24px}}
td,th{{border-bottom:1px solid #ddd;padding:6px;text-align:left}}h1{{font-size:20px}}
h2{{font-size:16px;margin-top:24px}}p{{color:#555;font-size:13px}}
tr.in td{{color:#555}}</style></head><body>
<h1>Gen318 Edge &middot; {_esc(_config.device_id if _config else '')}</h1>
<p>Works offline. Page refreshes every {PAGE_REFRESH_SECONDS} seconds. Times are WAT.</p>
<h2>SMS</h2><p>What the box has sent, newest first. Phone numbers are never shown or stored.</p>
<table><tr><th>Time</th><th>Who</th><th>Message</th></tr>
{messages or '<tr><td colspan="3">No SMS yet.</td></tr>'}</table>
<h2>Alerts</h2><p>To mark an alert, reply to its SMS: 1 = real, 2 = false alarm, 3 = other cause.</p>
<table><tr><th>Ref</th><th>Fault</th><th>Level</th><th>Raised</th><th>Status</th><th>Marked</th></tr>
{alerts or '<tr><td colspan="6">No alerts yet.</td></tr>'}</table>
<h2>What this box has learned</h2>
<p>Models train here from this site's own data: solar, load and battery run-out after 7 days, battery health after 14.</p>
<table><tr><th>Model</th><th>Trained</th><th>Days of data</th></tr>
{learned or '<tr><td colspan="3">Still collecting data.</td></tr>'}</table>
<h2>Alert sensitivity changed by your replies</h2>
<p>Three false-alarm replies in the last five make a warning less sensitive (two steps at most).
A reply of 1 (real), or RESET TUNING, puts it back. Critical alerts never change.</p>
<table><tr><th>Warning</th><th>Step</th><th>Installer set</th><th>Now</th></tr>
{tuning or '<tr><td colspan="4">Nothing changed: all warnings are at the installer settings.</td></tr>'}</table>
<h2>What this box itself uses</h2>
<p>{box_energy}<br>{box_models}</p>
<h2>Latest readings</h2>
<table><tr><th>Asset</th><th>Metric</th><th>Value</th></tr>
{rows or '<tr><td colspan="3">No readings yet.</td></tr>'}</table>
</body></html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
