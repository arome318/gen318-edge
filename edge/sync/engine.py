"""4-tier priority sync engine with connectivity-aware behavior.

Syncs edge data to cloud via HTTPS with gzip compression and device
certificate authentication. Adapts sync strategy based on detected
connectivity state (4G → full sync, 2G → alerts only, offline → queue).
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import math
import os
import ssl
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog
from common.database import EdgeDatabase
from common.model_manifest import parse_trusted_keys, verify_manifest

logger = structlog.get_logger()

# Applied on the asyncio loop by _apply_commands; SyncService supplies it.
CommandHandler = Callable[[list[dict[str, Any]]], Awaitable[None]]

# Cloud ingestion endpoints (device-authenticated via X-Device-Id; the
# registered edge_nodes row is the authority for site/tenant, so payloads
# do not carry them).
TELEMETRY_PATH = "/api/v1/ingest/telemetry"
EVENTS_PATH = "/api/v1/ingest/events"
ALERT_PATH = "/api/v1/ingest/alert"

# Server-side TelemetryBatchIn max_length — batches above this always 422.
TELEMETRY_MAX_ROWS_PER_POST = 10000

# Edge quality codes (0=good, 1=suspect, 2=bad) → cloud QualityLevel.
# Suspect maps to BAD (4) so questionable readings never pass as good;
# unknown codes are conservatively BAD too.
_EDGE_TO_CLOUD_QUALITY = {0: 0, 1: 4, 2: 4}

# Prune synced/rejected rows once per this many sync cycles (~hourly at the
# default 60s interval).
PRUNE_INTERVAL_CYCLES = 60


def _is_uuid(value: Any) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


_PEM_BEGIN = "-----BEGIN CERTIFICATE-----"
_PEM_END = "-----END CERTIFICATE-----"


def _local_cert_fingerprint(cert_path: str) -> str | None:
    """SHA-256 of the DER form of this device's certificate, 64 lowercase hex.

    Identical to what ``scripts/edge-pki.sh cert_fingerprint`` produces and to
    Caddy's ``{http.request.tls.client.fingerprint}``, so the cloud can compare
    it against ``edge_nodes.cert_fingerprint`` with plain string equality.

    Reporting ONLY: the cloud stores this in ``cert_fingerprint_reported`` and
    never authenticates on it. Its value is that it makes the mTLS cut-over
    pre-flightable — every node whose registration is a placeholder, or whose
    certificate was reissued after registration, becomes visible while
    enforcement is still off.

    Only the LEAF is hashed: openssl-issued device certs routinely carry a text
    dump above the PEM block and the issuing CA below it, and Caddy forwards
    the leaf's digest.

    Stdlib only (no new dependency on the Pi) and every failure degrades to
    ``None`` with a log, per the optional-dependency rule.
    """
    try:
        with open(cert_path) as fh:
            pem = fh.read()
        begin = pem.find(_PEM_BEGIN)
        end = pem.find(_PEM_END, begin)
        if begin == -1 or end == -1:
            logger.info("device_cert_fingerprint_unavailable", path=cert_path)
            return None
        der = ssl.PEM_cert_to_DER_cert(pem[begin : end + len(_PEM_END)] + "\n")
        return hashlib.sha256(der).hexdigest()
    except (OSError, ValueError):
        logger.info("device_cert_fingerprint_unavailable", path=cert_path)
        return None


class Priority(IntEnum):
    """Sync queue priority tiers."""

    CRITICAL = 0  # Alerts — immediate via MQTT QoS 2
    HIGH = 1  # Events — < 5 minutes
    NORMAL = 2  # Recent telemetry — < 15 minutes
    BACKFILL = 3  # Historical data — best-effort


class ConnectivityState(IntEnum):
    """Network connectivity classification."""

    OFFLINE = 0
    TWOG = 1
    THREEG = 2
    FOURG = 3
    WIFI = 4


@dataclass
class SyncConfig:
    """Configuration for the sync engine."""

    cloud_api_url: str
    device_id: str
    device_cert_path: str
    device_key_path: str
    ca_cert_path: str
    cert_pin_sha256: str
    mqtt_host: str = "mosquitto"
    mqtt_port: int = 1883
    sync_interval: int = 60
    backfill_max_batch_mb: int = 5
    p1_max_items_on_2g: int = 10
    # Telemetry drain (P2 recent / P3 backfill) — batch rows per POST and cap
    # batches per cycle so one cycle can't monopolize the link.
    telemetry_batch_size: int = 1000
    telemetry_max_batches_per_cycle: int = 10
    telemetry_recent_window_hours: int = 24
    # Synced/rejected rows older than this are pruned from SQLite.
    telemetry_retention_hours: int = 1080  # 45 days: on-site models look back 30
    model_path: str = "/models"
    model_trusted_public_keys_json: str = "{}"
    model_artifact_max_bytes: int = 50 * 1024 * 1024


class SyncEngine:
    """Connectivity-aware sync engine with 4-tier priority queue."""

    def __init__(self, config: SyncConfig, db: EdgeDatabase) -> None:
        self.config = config
        self.db = db
        self._connectivity = ConnectivityState.OFFLINE
        self._last_sync_at: float | None = None
        self._total_synced = 0
        self._wakeup = asyncio.Event()
        # Computed once: the certificate does not change under a running
        # container, and the heartbeat must not do file I/O every 60s.
        self._device_cert_fingerprint = _local_cert_fingerprint(config.device_cert_path)
        # Set by SyncService. Invoked on the loop with the cloud's heartbeat
        # commands; None in tests and in any deployment that does not wire it.
        self._on_commands: CommandHandler | None = None
        self._ota_check_requested = False
        self._trusted_model_keys = parse_trusted_keys(
            config.model_trusted_public_keys_json
        )
        self._telemetry_batch_size = min(
            config.telemetry_batch_size, TELEMETRY_MAX_ROWS_PER_POST
        )
        if config.telemetry_batch_size > TELEMETRY_MAX_ROWS_PER_POST:
            logger.warning(
                "telemetry_batch_size_clamped",
                configured=config.telemetry_batch_size,
                max=TELEMETRY_MAX_ROWS_PER_POST,
            )

        # Configure HTTPS client with device certificate + pinning (F-SEC-07)
        try:
            from sync.cert_pin import create_pinned_client

            self._http_client = create_pinned_client(
                pin_sha256=config.cert_pin_sha256,
                cert_path=config.device_cert_path,
                key_path=config.device_key_path,
                ca_cert_path=config.ca_cert_path,
                timeout=30.0,
            )
        except Exception:
            # Certs may not exist in dev/test — fall back to unverified
            logger.warning(
                "sync_cert_not_configured",
                msg="Using HTTP client without device certificates or pinning",
            )
            self._http_client = httpx.AsyncClient(
                verify=False,
                timeout=30.0,
            )

    async def check_connectivity(self) -> ConnectivityState:
        """Detect connectivity by measuring latency to cloud API.

        This is the fallback method. The ConnectivityDetector in
        connectivity.py provides modem-based detection as primary.
        """
        try:
            start = time.monotonic()
            resp = await self._http_client.get(
                f"{self.config.cloud_api_url}/health",
                timeout=10.0,
            )
            latency_ms = (time.monotonic() - start) * 1000

            if resp.status_code != 200:
                return ConnectivityState.OFFLINE

            if latency_ms < 200:
                return ConnectivityState.FOURG
            elif latency_ms < 800:
                return ConnectivityState.THREEG
            elif latency_ms < 3000:
                return ConnectivityState.TWOG
            else:
                return ConnectivityState.OFFLINE

        except (httpx.ConnectError, httpx.TimeoutException):
            return ConnectivityState.OFFLINE

    async def flush_queue(self, priority: Priority, max_items: int = 0) -> int:
        """Flush items of a given priority from sync queue. Returns count sent.

        Queue items are events (inference results, OTA status) and alerts —
        they are posted individually to the matching ingestion endpoint:
        P0 → /ingest/alert, everything else → /ingest/events. Telemetry does
        NOT ride this queue; it drains from measurements via flush_telemetry().

        Only validation rejections consume an item's dead-letter attempt
        budget. A transient outcome (throttling, 5xx, link failure) stops the
        drain — the queue is FIFO, so once the link or the rate budget is
        gone, the remaining posts this cycle are doomed too.
        """
        items = self.db.get_queue_items(
            priority=priority,
            limit=max_items or 1000,
        )
        sent = 0
        for item in items:
            outcome = await self._send_queue_item(item, priority)
            if outcome == "sent":
                sent += 1
            elif outcome == "rejected":
                self.db.increment_attempt(item["id"])
            else:  # transient
                break
        return sent

    async def _send_queue_item(self, item: dict[str, Any], priority: Priority) -> str:
        """Post one queue item; returns 'sent', 'rejected', or 'transient'."""
        payload = item["payload"]
        if priority == Priority.CRITICAL:
            path = ALERT_PATH
            body = {
                "title": str(
                    payload.get("title") or payload.get("type") or "Edge alert"
                )[:255],
                "severity": str(payload.get("severity") or "critical")[:20],
                "source": "edge",
                "payload": payload,
            }
        else:
            path = EVENTS_PATH
            event_type = payload.get("type") or (
                f"inference.{payload['model']}" if payload.get("model") else "edge.sync"
            )
            body = {
                "type": str(event_type)[:50],
                "severity": "informational",
                "source": "edge",
                "title": str(payload.get("title") or event_type)[:255],
                "payload": payload,
            }
            asset_id = payload.get("asset_id")
            # EventIn.asset_id must be a UUID; unmapped vendor ids stay
            # available inside body["payload"].
            if asset_id and _is_uuid(asset_id):
                body["asset_id"] = asset_id

        try:
            resp = await self._http_client.post(
                f"{self.config.cloud_api_url}{path}",
                json=body,
                headers={"X-Device-Id": self.config.device_id},
            )
        except httpx.RequestError as exc:
            logger.warning("sync_item_failed", error=str(exc))
            return "transient"

        if resp.status_code in (200, 201, 202):
            self.db.mark_synced([item["id"]])
            self._last_sync_at = time.time()
            self._total_synced += 1
            return "sent"
        if resp.status_code in (400, 422):
            # Genuine payload rejection — burns one of the item's 10 attempts.
            logger.warning(
                "sync_item_rejected",
                status=resp.status_code,
                attempts=item["attempts"],
            )
            return "rejected"
        # 401/403 (device not yet registered / temporarily blocked), 404,
        # 429, 5xx — device- or batch-level, says nothing about the payload.
        logger.warning("sync_item_deferred", status=resp.status_code)
        return "transient"

    # ── Telemetry drain (P2 recent / P3 backfill) ────────────────

    async def flush_telemetry(self, *, backfill: bool = False) -> int:
        """Drain unsynced measurements to the cloud in gzip'd batches.

        ``backfill=False`` drains only the recent window (P2, < 15 min
        target); ``backfill=True`` drains older rows (P3, best-effort on
        good links). Stops early on transient failure — rows stay unsynced
        and retry next cycle.
        """
        cutoff_ms = int(
            (time.time() - self.config.telemetry_recent_window_hours * 3600) * 1000
        )
        since_ms = None if backfill else cutoff_ms
        before_ms = cutoff_ms if backfill else None

        total = 0
        for _ in range(self.config.telemetry_max_batches_per_cycle):
            rows = self.db.get_unsynced_measurements(
                limit=self._telemetry_batch_size,
                since_ms=since_ms,
                before_ms=before_ms,
            )
            if not rows:
                break
            handled = await self._post_measurements(rows)
            if handled == 0:
                break  # transient failure — retry next cycle
            total += handled
        return total

    @staticmethod
    def _measurement_keys(
        rows: list[dict[str, Any]],
    ) -> tuple[list[int], list[str], list[str]]:
        return (
            [row["timestamp"] for row in rows],
            [row["asset_id"] for row in rows],
            [row["metric_name"] for row in rows],
        )

    async def _post_measurements(self, rows: list[dict[str, Any]]) -> int:
        """POST one measurement batch; returns rows handled (synced or rejected).

        Rows that can never validate cloud-side — non-UUID asset_id (missing
        asset_mappings entry) or non-finite value — are marked rejected up
        front without burning HTTP requests. Cloud accept (202) marks the
        rest synced. A 422 bisects the batch to isolate poison rows. Every
        other outcome (401/403 during the registration window, 404, 413,
        429, 5xx, transport errors) is treated as transient: rows stay
        unsynced and retry next cycle — NEVER tombstoned.
        """
        sendable: list[dict[str, Any]] = []
        local_poison: list[dict[str, Any]] = []
        for row in rows:
            if _is_uuid(row["asset_id"]) and math.isfinite(row["value"]):
                sendable.append(row)
            else:
                local_poison.append(row)
        if local_poison:
            logger.warning(
                "telemetry_rows_unmappable",
                count=len(local_poison),
                sample_asset=str(local_poison[0]["asset_id"]),
                sample_metric=local_poison[0]["metric_name"],
            )
            self.db.mark_measurements_rejected(*self._measurement_keys(local_poison))
        if not sendable:
            return len(local_poison)

        # The batch identity is derived from the immutable SQLite primary keys,
        # so a transport retry has the same provenance even across process
        # restarts. A size split gets its own stable child identity.
        identity = "|".join(
            f"{row['timestamp']}:{row['asset_id']}:{row['metric_name']}"
            for row in sendable
        )
        batch_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"gen318:telemetry:{self.config.device_id}:{identity}",
            )
        )
        body = json.dumps(
            {
                "batch_id": batch_id,
                "schema_version": 1,
                "sent_at": datetime.now(UTC).isoformat(),
                "measurements": [
                    {
                        "time": datetime.fromtimestamp(
                            row["timestamp"] / 1000, tz=UTC
                        ).isoformat(),
                        "asset_id": row["asset_id"],
                        "metric_name": row["metric_name"],
                        "value": row["value"],
                        "quality": _EDGE_TO_CLOUD_QUALITY.get(row["quality"], 4),
                        "sequence_number": sequence,
                        "metadata": {"edge_quality": row["quality"]},
                    }
                    for sequence, row in enumerate(sendable)
                ],
            }
        ).encode()
        compressed = gzip.compress(body, compresslevel=6)

        # Cap batch bytes on the wire (P3 ≤ 5 MB) — size-split
        # oversized batches before sending.
        max_mb = self.config.backfill_max_batch_mb
        if max_mb and len(compressed) > max_mb * 1024 * 1024 and len(sendable) > 1:
            mid = len(sendable) // 2
            handled = await self._post_measurements(sendable[:mid])
            handled += await self._post_measurements(sendable[mid:])
            return handled + len(local_poison) if handled else 0

        try:
            resp = await self._http_client.post(
                f"{self.config.cloud_api_url}{TELEMETRY_PATH}",
                content=compressed,
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                    "X-Device-Id": self.config.device_id,
                },
            )
        except httpx.RequestError as exc:
            logger.warning("telemetry_sync_failed", error=str(exc))
            return 0

        keys = self._measurement_keys(sendable)
        if resp.status_code in (200, 201, 202):
            self.db.mark_measurements_synced(*keys)
            self._last_sync_at = time.time()
            self._total_synced += len(sendable)
            logger.info(
                "telemetry_batch_sent",
                count=len(sendable),
                size_kb=round(len(compressed) / 1024, 1),
            )
            return len(sendable) + len(local_poison)

        if resp.status_code == 422:
            # Row-level validation rejection — bisect to isolate the poison
            # row(s) instead of dropping (or endlessly retrying) the batch.
            if len(sendable) == 1:
                logger.warning(
                    "telemetry_row_rejected",
                    status=resp.status_code,
                    asset_id=sendable[0]["asset_id"],
                    metric=sendable[0]["metric_name"],
                )
                self.db.mark_measurements_rejected(*keys)
                return 1 + len(local_poison)
            mid = len(sendable) // 2
            handled = await self._post_measurements(sendable[:mid])
            handled += await self._post_measurements(sendable[mid:])
            return handled + len(local_poison) if handled else 0

        # 401/403/404/413/429/5xx — batch- or device-level, not row-level.
        logger.warning(
            "telemetry_batch_deferred", status=resp.status_code, count=len(sendable)
        )
        return 0

    def _read_watchdog_status(self) -> dict[str, Any]:
        """Read system status from shared volume written by watchdog.

        The watchdog container writes /data/metrics/system.json atomically
        with container_status, system_metrics, and updated_at timestamp.
        Falls back to empty dicts if file is missing or stale (>120s).
        """
        import json
        import os

        status_path = "/data/metrics/system.json"
        max_age_seconds = 120

        try:
            if not os.path.exists(status_path):
                return {"container_status": {}, "system_metrics": {}}

            with open(status_path) as f:
                data = json.load(f)

            # Check staleness
            updated_at = data.get("updated_at", 0)
            if time.time() - updated_at > max_age_seconds:
                logger.debug("watchdog_status_stale", age_s=time.time() - updated_at)
                return {"container_status": {}, "system_metrics": {}}

            return {
                "container_status": data.get("container_status", {}),
                "system_metrics": data.get("system_metrics", {}),
            }
        except (json.JSONDecodeError, OSError):
            return {"container_status": {}, "system_metrics": {}}

    # Cloud heartbeat vocabulary (backend/app/schemas/edge.py validates against
    # exactly this set). The local enum member names ("twog"/"threeg"/"fourg")
    # are NOT accepted — sending them 422s every heartbeat, which leaves the
    # node pinned at status='provisioning' forever while telemetry keeps
    # flowing, i.e. a silent monitoring failure. Map at the boundary; keep the
    # enum itself untouched so the tier-degradation logic is unaffected.
    _CLOUD_CONNECTIVITY = {
        ConnectivityState.OFFLINE: "offline",
        ConnectivityState.TWOG: "2g",
        ConnectivityState.THREEG: "3g",
        ConnectivityState.FOURG: "4g",
        ConnectivityState.WIFI: "4g",  # cloud vocabulary has no "wifi" member
    }

    async def send_heartbeat(self) -> dict[str, Any] | None:
        """Send heartbeat to cloud with real container + system data.

        Returns the parsed response body, or ``None`` on any transport error or
        non-2xx — which is exactly the old ``return False``, so the offline path
        is unchanged. The body is read because the heartbeat RESPONSE is the
        only cloud→device channel in a strictly pull-only design; see
        ``_apply_commands``.
        """
        model_state = self.db.get_model_state()
        model_versions = {m["model_name"]: m["version"] for m in model_state}

        # Read real metrics from watchdog's shared status file
        watchdog_data = self._read_watchdog_status()

        system_metrics = dict(watchdog_data["system_metrics"])
        storage_status_raw = self.db.get_config("storage.status")
        if storage_status_raw:
            try:
                storage_status = json.loads(storage_status_raw)
            except json.JSONDecodeError:
                storage_status = None
            if isinstance(storage_status, dict):
                # These become ordinary system_metrics on the cloud, so the
                # existing heartbeat contract and offline behavior stay intact.
                system_metrics.update(
                    storage_pressure=storage_status.get("pressure"),
                    storage_runway_days=storage_status.get("estimated_runway_days"),
                    storage_target_days=storage_status.get("target_days"),
                    storage_target_met=storage_status.get("target_met"),
                    storage_database_bytes=storage_status.get("database_bytes"),
                )

        body: dict[str, Any] = {
            "connectivity": self._CLOUD_CONNECTIVITY.get(self._connectivity, "unknown"),
            "sync_lag_seconds": int(time.time() - self._last_sync_at)
            if self._last_sync_at
            else 0,
            "container_status": watchdog_data["container_status"],
            "system_metrics": system_metrics,
            "model_versions": model_versions,
        }
        # Omitted entirely when no certificate is readable (dev/test), so an
        # older cloud and a cert-less node both see exactly today's payload.
        if self._device_cert_fingerprint:
            body["device_cert_fingerprint"] = self._device_cert_fingerprint
        # What we ACTUALLY run — the same key _on_model_update stamps. Until
        # this arrives the cloud's firmware_version is a wish the device has no
        # way to correct.
        firmware_version = self.db.get_config("firmware_version")
        if firmware_version:
            body["firmware_version"] = firmware_version

        try:
            resp = await self._http_client.post(
                f"{self.config.cloud_api_url}/api/v1/edge/nodes/"
                f"{self.config.device_id}/heartbeat",
                json=body,
                headers={"X-Device-Id": self.config.device_id},
            )
            if resp.status_code not in (200, 201, 202):
                return None
            try:
                parsed = resp.json()
            except ValueError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        except (httpx.ConnectError, httpx.TimeoutException):
            return None

    async def _apply_commands(self, response: dict[str, Any] | None) -> None:
        """Hand any cloud commands to the service-level handler.

        Wrapped in its own try/except: a command failure must never break a
        sync cycle, and a cycle that dies takes the telemetry drain with it.
        Commands are applied on the asyncio loop — no MQTT callback thread is
        involved, so the known ``create_task``-from-callback violation in
        ``sync/main.py::_on_ssh_command`` is not copied here.
        """
        if not response or self._on_commands is None:
            return
        commands = response.get("commands")
        if not isinstance(commands, list) or not commands:
            return
        for command in commands:
            if isinstance(command, dict) and command.get("type") == "ota_check":
                # A hint, not a push. Ignoring it costs time and nothing else.
                self._ota_check_requested = True
        try:
            await self._on_commands([c for c in commands if isinstance(c, dict)])
        except Exception:
            logger.exception("cloud_command_handling_failed")

    OTA_CHECK_INTERVAL_CYCLES = 360  # Every 360 cycles × 60s = 6 hours

    def _record_model_download_failure(
        self, manifest: dict[str, Any], reason: str
    ) -> None:
        self.db.enqueue_sync(
            priority=1,
            payload={
                "type": "ota_status",
                "device_id": self.config.device_id,
                "target_version": manifest.get("version", ""),
                "models_accepted": 0,
                "models_total": 1,
                "model_results": {manifest.get("model_name", "unknown"): reason},
                "status": "download_failed",
                "timestamp": int(time.time() * 1000),
            },
        )

    async def _download_model(self, manifest: dict[str, Any]) -> dict[str, Any] | None:
        verified, reason = verify_manifest(manifest, self._trusted_model_keys)
        if not verified:
            self._record_model_download_failure(manifest, reason)
            logger.warning("model_manifest_rejected", reason=reason)
            return None

        targets = manifest.get("target_nodes")
        if not isinstance(targets, list) or not (
            "*" in targets or self.config.device_id in targets
        ):
            self._record_model_download_failure(manifest, "wrong_target")
            return None
        size_bytes = manifest.get("size_bytes")
        if (
            not isinstance(size_bytes, int)
            or size_bytes <= 0
            or size_bytes > self.config.model_artifact_max_bytes
        ):
            self._record_model_download_failure(manifest, "invalid_size")
            return None
        download_url = manifest.get("download_url")
        if (
            not isinstance(download_url, str)
            or urlparse(download_url).scheme != "https"
        ):
            self._record_model_download_failure(manifest, "insecure_download_url")
            return None

        staging = Path(self.config.model_path) / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        safe_model = (
            "".join(
                char
                for char in str(manifest.get("model_name", "model"))
                if char.isalnum() or char in "-_"
            )
            or "model"
        )
        destination = staging / f"{manifest['update_id']}-{safe_model}.ubj"
        partial = destination.with_suffix(".partial")
        digest = hashlib.sha256()
        received = 0
        try:
            # A separate client is intentional: pre-signed object URLs must
            # never receive the device's private client-certificate identity.
            async with httpx.AsyncClient(
                timeout=60.0, follow_redirects=False
            ) as client:
                async with client.stream("GET", download_url) as response:
                    if response.status_code != 200:
                        raise ValueError(f"http_{response.status_code}")
                    with partial.open("wb") as artifact:
                        async for chunk in response.aiter_bytes():
                            received += len(chunk)
                            if (
                                received > size_bytes
                                or received > self.config.model_artifact_max_bytes
                            ):
                                raise ValueError("oversize")
                            digest.update(chunk)
                            artifact.write(chunk)
                        artifact.flush()
                        os.fsync(artifact.fileno())
            if received != size_bytes:
                raise ValueError("size_mismatch")
            if digest.hexdigest() != manifest.get("sha256"):
                raise ValueError("sha256_mismatch")
            os.replace(partial, destination)
        except (httpx.HTTPError, OSError, ValueError) as exc:
            partial.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            self._record_model_download_failure(manifest, str(exc))
            logger.warning("model_artifact_download_failed", error=str(exc))
            return None

        return {**manifest, "path": str(destination)}

    async def check_for_ota(self, mqtt_client: Any = None) -> dict[str, Any] | None:
        """Poll cloud for pending OTA updates. Publish to local MQTT if found.

        Called every 6 hours from sync_loop. The inference container
        subscribes to TOPIC_MODEL_UPDATES and applies the update.
        """
        current_version = self.db.get_config("firmware_version") or ""
        current_models = {
            row["model_name"]: row["version"]
            for row in self.db.get_model_state()
            if row.get("model_name") and row.get("version")
        }
        try:
            resp = await self._http_client.get(
                f"{self.config.cloud_api_url}/api/v1/edge/nodes/"
                f"{self.config.device_id}/updates/pending",
                params={
                    "current_version": current_version,
                    "current_models": json.dumps(current_models, separators=(",", ":")),
                },
                headers={"X-Device-Id": self.config.device_id},
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            if data.get("update_available"):
                prepared_models: list[dict[str, Any]] = []
                raw_models = data.get("models", [])
                if isinstance(raw_models, list):
                    for manifest in raw_models:
                        if isinstance(manifest, dict):
                            downloaded = await self._download_model(manifest)
                            if downloaded is not None:
                                prepared_models.append(downloaded)
                            else:
                                # Keep failed manifests in the local contract.
                                # Inference counts/rejects them again, so a
                                # partial download can never look like a fully
                                # applied firmware bundle and stamp success.
                                prepared_models.append(manifest)
                logger.info(
                    "ota_update_available",
                    target=data["target_version"],
                    channel=data["channel"],
                )
                # Publish to local MQTT for inference container
                if mqtt_client:
                    from common.mqtt_client import TOPIC_MODEL_UPDATES

                    mqtt_client.publish(
                        TOPIC_MODEL_UPDATES,
                        {
                            "action": "update",
                            "target_version": data["target_version"],
                            "channel": data["channel"],
                            "device_id": self.config.device_id,
                            "firmware_update": data.get("firmware_update", True),
                            "models": prepared_models,
                        },
                    )
                return data
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        return None

    def request_flush(self) -> None:
        """Wake the sync loop early (e.g. a P0 alert just queued).

        Must be called from the event-loop thread; from MQTT/GPIO callback
        threads, bridge via loop.call_soon_threadsafe(engine.request_flush).
        """
        self._wakeup.set()

    async def sync_loop(self, mqtt_client: Any = None) -> None:
        """Main sync loop — runs continuously with connectivity-aware behavior."""
        cycle = 0

        while True:
            cycle += 1
            try:
                await self._sync_cycle(cycle, mqtt_client)
            except Exception:
                # One bad cycle (unexpected transport/sqlite error) must never
                # kill the service — the watchdog restart is a last resort,
                # not the retry mechanism.
                logger.exception("sync_cycle_failed", cycle=cycle)

            # Interruptible sleep: request_flush() cuts it short so P0 alerts
            # don't wait out a full interval. Offline is still safe — the next
            # cycle re-checks connectivity and queues everything locally.
            try:
                await asyncio.wait_for(
                    self._wakeup.wait(), timeout=self.config.sync_interval
                )
            except TimeoutError:
                pass
            self._wakeup.clear()

    async def _sync_cycle(self, cycle: int, mqtt_client: Any = None) -> None:
        """One connectivity-aware sync pass."""
        self._connectivity = await self.check_connectivity()

        logger.info(
            "sync_cycle",
            cycle=cycle,
            connectivity=self._connectivity.name,
            queue=self.db.get_queue_depth_by_priority(),
            unsynced_measurements=self.db.count_unsynced_measurements(),
        )

        # Retention sweep (local, runs regardless of connectivity). Never
        # touches synced=0 rows — offline backlog is kept until confirmed.
        if cycle % PRUNE_INTERVAL_CYCLES == 0:
            cutoff_ms = int(
                (time.time() - self.config.telemetry_retention_hours * 3600) * 1000
            )
            pruned = self.db.prune_synced_measurements(cutoff_ms)
            if pruned:
                logger.info("measurements_pruned", count=pruned)

        # Heartbeat right after the P0 flush on connected states: it is one
        # small POST, and its cadence must not starve behind large drains.
        if self._connectivity >= ConnectivityState.FOURG:
            # Full sync: all priorities, incl. telemetry backfill
            await self.flush_queue(Priority.CRITICAL)
            await self._apply_commands(await self.send_heartbeat())
            await self.flush_queue(Priority.HIGH)
            await self.flush_queue(Priority.NORMAL)
            await self.flush_queue(Priority.BACKFILL)
            await self.flush_telemetry(backfill=False)
            await self.flush_telemetry(backfill=True)

            # Check for OTA updates every 6 hours, or off-cycle when the cloud
            # hinted (halt/rollback propagation drops from 6h to ~60s). The
            # interval and the tier ladder are unchanged.
            if cycle % self.OTA_CHECK_INTERVAL_CYCLES == 0 or self._ota_check_requested:
                self._ota_check_requested = False
                await self.check_for_ota(mqtt_client)

        elif self._connectivity >= ConnectivityState.THREEG:
            await self.flush_queue(Priority.CRITICAL)
            await self._apply_commands(await self.send_heartbeat())
            await self.flush_queue(Priority.HIGH)
            await self.flush_queue(Priority.NORMAL)
            sent = await self.flush_telemetry(backfill=False)
            cap = (
                self._telemetry_batch_size * self.config.telemetry_max_batches_per_cycle
            )
            if sent < cap:
                # Recent window fully drained this cycle — spend leftover
                # budget on >24h backlog so slow links never strand it.
                await self.flush_telemetry(backfill=True)

        elif self._connectivity >= ConnectivityState.TWOG:
            # Critical + limited high
            await self.flush_queue(Priority.CRITICAL)
            await self.flush_queue(
                Priority.HIGH,
                max_items=self.config.p1_max_items_on_2g,
            )

        # Offline: everything queues locally — no action needed

    @property
    def connectivity(self) -> ConnectivityState:
        return self._connectivity

    @property
    def last_sync_at(self) -> float | None:
        return self._last_sync_at

    @property
    def total_synced(self) -> int:
        return self._total_synced

    async def close(self) -> None:
        """Clean up HTTP client."""
        await self._http_client.aclose()
