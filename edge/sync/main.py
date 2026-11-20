"""Edge sync service — main entry point.

Runs as the gen318-sync container. Coordinates the sync engine,
connectivity detector, and reverse SSH tunnel manager.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import structlog
from common.config import EdgeConfig, load_config
from common.database import EdgeDatabase
from common.mqtt_client import (
    TOPIC_ALERTS_CRITICAL,
    TOPIC_CONTROL_SSH,
    MQTTClient,
)

from sync.engine import Priority, SyncConfig, SyncEngine
from sync.reverse_ssh import SSH_GRANT_CONFIG_KEY, ReverseSSHTunnel

logger = structlog.get_logger()


def _grant_token_proof(grant_id: str, token: Any) -> str | None:
    """Proof of possession of a grant token, echoed instead of the token.

    MIRRORED IN THE CLOUD at
    ``backend/app/services/edge_access_service.py::derive_token_proof`` — the
    edge may never import from ``backend/``, so the formula is duplicated
    deliberately. Change both or neither.

    Why not just send the token: this report goes into the P1 sync queue and
    out through ``POST /ingest/events``, where the cloud's critical
    ``events_raw_handler`` writes the body verbatim into ``event_outbox`` and
    ``events``. Echoing the token would leave a plaintext shell credential in
    two cloud tables for the life of those rows. The proof still requires
    holding the token, and is bound to this one grant.
    """
    if not token or not isinstance(token, str):
        return None
    return hashlib.sha256(f"{token}:{grant_id}".encode()).hexdigest()


class SyncService:
    """Orchestrates cloud synchronization and remote access."""

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self.db = EdgeDatabase(config.sqlite_path, config.sqlite_encryption_key)
        self.mqtt = MQTTClient(
            host=config.mqtt_host,
            port=config.mqtt_port,
            client_id="gen318-sync",
        )

        sync_config = SyncConfig(
            cloud_api_url=config.cloud_api_url,
            device_id=config.device_id,
            device_cert_path=config.device_cert_path,
            device_key_path=config.device_key_path,
            ca_cert_path=config.ca_cert_path,
            cert_pin_sha256=config.cert_pin_sha256,
            mqtt_host=config.mqtt_host,
            mqtt_port=config.mqtt_port,
            sync_interval=config.sync_interval_seconds,
            backfill_max_batch_mb=config.backfill_max_batch_mb,
            p1_max_items_on_2g=config.p1_max_items_on_2g,
            telemetry_batch_size=config.telemetry_batch_size,
            telemetry_max_batches_per_cycle=config.telemetry_max_batches_per_cycle,
            telemetry_recent_window_hours=config.telemetry_recent_window_hours,
            telemetry_retention_hours=config.telemetry_retention_hours,
            model_path=config.model_path,
            model_trusted_public_keys_json=config.model_trusted_public_keys_json,
            model_artifact_max_bytes=config.model_artifact_max_bytes,
        )

        self.engine = SyncEngine(sync_config, self.db)
        self.ssh_tunnel = ReverseSSHTunnel(
            device_id=config.device_id,
            device_key_path=config.device_key_path,
            db=self.db,
        )
        # The cloud→device channel: commands ride the heartbeat response.
        self.engine._on_commands = self._handle_cloud_commands
        self._loop: asyncio.AbstractEventLoop | None = None

    def _on_critical_alert(self, payload: dict[str, Any]) -> None:
        """Bridge watchdog tamper/critical alerts into the P0 sync queue.

        Fires on the paho-MQTT thread: enqueue_sync is thread-safe SQLite
        (lock + check_same_thread=False), so only the loop wakeup needs
        call_soon_threadsafe. Offline-safe — the queue drains when the link
        returns.
        """
        try:
            self.db.enqueue_sync(priority=Priority.CRITICAL, payload=payload)
        except Exception:
            logger.exception("critical_alert_enqueue_failed")
            return

        logger.info("critical_alert_queued", type=payload.get("type"))
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.engine.request_flush)

    async def _handle_cloud_commands(self, commands: list[dict[str, Any]]) -> None:
        """Apply commands returned in the cloud's heartbeat response.

        Runs on the asyncio loop (called from ``SyncEngine._apply_commands``),
        so no thread bridging is needed and the known
        ``create_task``-from-MQTT-callback violation below is NOT copied.

        Offline-safe: the outcome goes into the LOCAL P1 sync queue and is
        drained later by the normal tier ladder. Nothing here awaits the cloud,
        so a link that drops mid-command costs a delay, not the report.
        """
        for command in commands:
            action = command.get("type")
            grant_id = command.get("grant_id")
            if action == "open_tunnel" and grant_id:
                await self._open_granted_tunnel(command)
            elif action == "close_tunnel" and grant_id:
                await self.ssh_tunnel.close()
                self._report_grant(str(grant_id), "closed", command.get("token"))
            elif action == "ota_check":
                # Consumed by the engine itself (it sets the off-cycle flag).
                continue
            elif action in {"dispatch_command", "dispatch_cancel"}:
                # Gen318 Edge is advisory-only: it never writes to equipment.
                logger.warning("dispatch_command_ignored", type=action)

    async def _open_granted_tunnel(self, command: dict[str, Any]) -> None:
        grant_id = str(command.get("grant_id"))
        token = command.get("token")
        try:
            result = await self.ssh_tunnel.open(
                remote_port=int(command.get("remote_port", 22222)),
                expires_at=command.get("expires_at"),
                grant_id=grant_id,
                # Persisted with the grant so a watchdog restart can still
                # report it closed. Never the token itself.
                token_proof=_grant_token_proof(grant_id, token),
            )
        except Exception:
            logger.exception("ssh_tunnel_open_failed", grant_id=grant_id)
            self._report_grant(grant_id, "failed", token)
            return
        status = (
            "opened" if result["status"] in ("opened", "already_open") else "failed"
        )
        self._report_grant(grant_id, status, token)

    def _report_grant(self, grant_id: str, status: str, token: Any) -> None:
        """Queue the outcome (P1) with a PROOF, never the token itself.

        The cloud verifies the proof before acting, so a report without one is
        dropped there rather than trusted. The token stays in memory on this
        device: everything queued here is persisted verbatim in the cloud's
        `event_outbox` and `events` tables.
        """
        self._enqueue_grant_report(
            grant_id, status, _grant_token_proof(grant_id, token)
        )

    def _enqueue_grant_report(
        self, grant_id: str, status: str, proof: str | None
    ) -> None:
        payload: dict[str, Any] = {
            "type": "edge.access_grant",
            "device_id": self.config.device_id,
            "grant_id": grant_id,
            "status": status,
        }
        if proof:
            payload["token_proof"] = proof
        try:
            self.db.enqueue_sync(priority=Priority.HIGH, payload=payload)
        except Exception:
            logger.exception("grant_report_enqueue_failed", grant_id=grant_id)

    def _recover_orphaned_grant(self) -> None:
        """Close the books on a grant this container was restarted out of.

        The watchdog restarts containers routinely and everything inside dies
        with the PID namespace — the ssh child and the in-process auto-close
        task both go with it. What does NOT die is the cloud's belief that the
        tunnel is `active`, and for a remote-shell audit trail a record that
        disagrees with the hardware is the one thing it must never be.

        So: the tunnel is already gone by definition (nothing to reap), the
        persisted record is the only memory of it, and the honest action is to
        report it closed and forget it. Offline-safe — the report goes into the
        local P1 queue and drains whenever the link returns. Never raises: a
        failure here must not stop the sync service starting.
        """
        try:
            raw = self.db.get_config(SSH_GRANT_CONFIG_KEY)
            # Not a string means no usable record — an unconfigured key, or a
            # stubbed database in a test. Leave it alone rather than guess.
            if not raw or not isinstance(raw, str):
                return
            try:
                record = json.loads(raw)
            except ValueError:
                record = {}
            grant_id = record.get("grant_id") if isinstance(record, dict) else None
            if not grant_id:
                self._clear_grant_record()
                return
            logger.warning(
                "ssh_grant_orphaned_by_restart",
                grant_id=grant_id,
                remote_port=record.get("remote_port"),
            )
            self._enqueue_grant_report(
                str(grant_id), "closed", record.get("token_proof")
            )
            self._clear_grant_record()
        except Exception:
            # Offline-first rule 1: nothing on the startup path may fail hard.
            logger.warning("ssh_grant_recovery_failed", exc_info=True)

    def _clear_grant_record(self) -> None:
        try:
            self.db.set_config(SSH_GRANT_CONFIG_KEY, "")
        except Exception:
            logger.warning("ssh_grant_recovery_clear_failed")

    def _on_ssh_command(self, payload: dict[str, Any]) -> None:
        """Handle SSH tunnel control commands from the LOCAL MQTT broker.

        Left in place untouched: it is the on-site operator's escape hatch, and
        it is the one known ``create_task``-from-a-paho-callback violation in
        the repo. The cloud path deliberately does not go through here.
        """
        action = payload.get("action", "")
        remote_port = payload.get("remote_port", 0)
        timeout_minutes = payload.get("timeout_minutes", 30)

        if self._loop is None:
            logger.error("ssh_command_no_loop")
            return

        if action == "open" and remote_port:
            self._loop.create_task(self.ssh_tunnel.open(remote_port, timeout_minutes))
        elif action == "close":
            self._loop.create_task(self.ssh_tunnel.close())
        else:
            logger.warning("ssh_unknown_command", action=action)

    async def run(self) -> None:
        """Start the sync service.

        Standalone mode: with no CLOUD_API_URL the box never contacts any
        server. Everything else (collection, detection, SMS alerts, local
        dashboard) works without this service.
        """
        if not self.config.cloud_api_url:
            logger.info("sync_standalone_mode", reason="CLOUD_API_URL not set")
            while True:
                await asyncio.sleep(3600)
        self._loop = asyncio.get_running_loop()
        self.db.connect()
        self.db.init_schema()
        # Before anything else touches the network: if this container was
        # restarted while a grant was live, the tunnel died with it and the
        # cloud still believes it is open.
        self._recover_orphaned_grant()
        self.mqtt.connect()

        # Subscribe to SSH control commands
        self.mqtt.subscribe(TOPIC_CONTROL_SSH, self._on_ssh_command)

        # Bridge watchdog tamper/critical alerts (published QoS 2) into the
        # P0 sync queue — without this they never leave the site.
        self.mqtt.subscribe(TOPIC_ALERTS_CRITICAL, self._on_critical_alert, qos=2)

        logger.info(
            "sync_service_started",
            device_id=self.config.device_id,
            cloud_url=self.config.cloud_api_url,
            interval_s=self.config.sync_interval_seconds,
        )

        try:
            await self.engine.sync_loop(mqtt_client=self.mqtt)
        finally:
            await self.engine.close()

    def stop(self) -> None:
        self.mqtt.disconnect()
        self.db.close()
        logger.info("sync_service_stopped")


async def main() -> None:
    config = load_config()
    service = SyncService(config)
    try:
        await service.run()
    except KeyboardInterrupt:
        service.stop()


if __name__ == "__main__":
    asyncio.run(main())
