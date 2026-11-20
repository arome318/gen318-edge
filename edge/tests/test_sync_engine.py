"""SyncEngine unit tests — priority queue, connectivity, heartbeat."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sync.engine import (
    ConnectivityState,
    Priority,
    SyncConfig,
    SyncEngine,
    _local_cert_fingerprint,
)

# Cloud MeasurementIn requires UUID asset ids (asset_mappings map vendor ids
# to Gen318 asset UUIDs before rows reach the sync engine).
ASSET_A = str(uuid.uuid4())
ASSET_B = str(uuid.uuid4())
ASSET_POISON = str(uuid.uuid4())

_PEM_BEGIN_MARKER = "-----BEGIN CERTIFICATE-----"
_PEM_END_MARKER = "-----END CERTIFICATE-----"

# A real EC P-256 clientAuth certificate, issued the same way
# scripts/edge-pki.sh issues device certs. Its openssl SHA-256 fingerprint is
# e5c8af04d5eecb82d094c9749103f357057ab2bde10808e89f9e25291a6933ef.
_SELF_SIGNED_PEM = """-----BEGIN CERTIFICATE-----
MIIBoTCCAUagAwIBAgIUBkBpv7UZo7XRH0UrWet5wXR6s04wCgYIKoZIzj0EAwIw
GzEZMBcGA1UEAwwQZ2VuMzE4LWVkZ2UtdGVzdDAeFw0yNjA4MDkwMzE2MjlaFw0z
NjA4MDYwMzE2MjlaMBsxGTAXBgNVBAMMEGdlbjMxOC1lZGdlLXRlc3QwWTATBgcq
hkjOPQIBBggqhkjOPQMBBwNCAASgz2K7zkMjv8NCP6WO/EqK9Fy5d1y2vMO3ftMQ
GN7RaJkT8NmPr0sPGIRCRBSJaNbpk40vVcnEqxQyN3GWEwxJo2gwZjAdBgNVHQ4E
FgQURa1pQO4ego/mMerD+GrQzGP5ht8wHwYDVR0jBBgwFoAURa1pQO4ego/mMerD
+GrQzGP5ht8wDwYDVR0TAQH/BAUwAwEB/zATBgNVHSUEDDAKBggrBgEFBQcDAjAK
BggqhkjOPQQDAgNJADBGAiEA3tDBpZC7zwvPByw78MoXzMV/aqk2IDPomH8J8g50
30QCIQCgOo8c/amqAM02mSiqSiqkD357YXGkwQAy+unZ48Ew8g==
-----END CERTIFICATE-----
"""


def _expected_fingerprint() -> str:
    """SHA-256 of the DER bytes, computed WITHOUT the ssl module.

    Deliberately an independent path from the implementation: base64-decoding
    the PEM body is what `openssl x509 -fingerprint -sha256` and Caddy's
    `{http.request.tls.client.fingerprint}` both hash. If the two ever diverge,
    the cloud comparison would silently never match.
    """
    body = "".join(
        line for line in _SELF_SIGNED_PEM.splitlines() if not line.startswith("-----")
    )
    return hashlib.sha256(base64.b64decode(body)).hexdigest()


@pytest.fixture
def sync_config() -> SyncConfig:
    return SyncConfig(
        cloud_api_url="https://api.gen318.io",
        device_id="test-edge-001",
        device_cert_path="/tmp/fake.pem",
        device_key_path="/tmp/fake.key",
        ca_cert_path="/tmp/fake-ca.pem",
        cert_pin_sha256="",
        sync_interval=1,
    )


@pytest.fixture
def sync_engine(sync_config, edge_db):
    engine = SyncEngine(sync_config, edge_db)
    # Replace HTTP client with a mock
    engine._http_client = AsyncMock()
    return engine


class TestConnectivity:
    @pytest.mark.asyncio
    async def test_4g_on_fast_response(self, sync_engine) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        sync_engine._http_client.get = AsyncMock(return_value=mock_resp)

        # Patch time.monotonic to simulate fast response
        with patch("sync.engine.time") as mock_time:
            mock_time.monotonic.side_effect = [0.0, 0.1]  # 100ms
            state = await sync_engine.check_connectivity()
            assert state == ConnectivityState.FOURG

    @pytest.mark.asyncio
    async def test_offline_on_timeout(self, sync_engine) -> None:
        import httpx

        sync_engine._http_client.get = AsyncMock(
            side_effect=httpx.TimeoutException("timeout")
        )
        state = await sync_engine.check_connectivity()
        assert state == ConnectivityState.OFFLINE


class TestFlushQueue:
    @pytest.mark.asyncio
    async def test_flush_posts_queue_item_as_event(self, sync_engine, edge_db) -> None:
        edge_db.enqueue_sync(
            1, {"model": "battery_health", "asset_id": "asset-1", "soh_pct": 91.2}
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 202
        sync_engine._http_client.post = AsyncMock(return_value=mock_resp)

        sent = await sync_engine.flush_queue(Priority.HIGH)
        assert sent == 1

        call = sync_engine._http_client.post.call_args
        assert call.args[0].endswith("/api/v1/ingest/events")
        assert call.kwargs["headers"]["X-Device-Id"] == "test-edge-001"
        body = call.kwargs["json"]
        assert body["type"] == "inference.battery_health"
        assert body["source"] == "edge"
        assert body["payload"]["soh_pct"] == 91.2
        # site/tenant omitted — the device registration is the authority
        assert "site_id" not in body
        assert "tenant_id" not in body

    @pytest.mark.asyncio
    async def test_flush_posts_critical_as_alert(self, sync_engine, edge_db) -> None:
        edge_db.enqueue_sync(0, {"type": "tamper", "severity": "critical"})

        mock_resp = MagicMock()
        mock_resp.status_code = 202
        sync_engine._http_client.post = AsyncMock(return_value=mock_resp)

        sent = await sync_engine.flush_queue(Priority.CRITICAL)
        assert sent == 1

        call = sync_engine._http_client.post.call_args
        assert call.args[0].endswith("/api/v1/ingest/alert")
        assert call.kwargs["json"]["severity"] == "critical"

        remaining = edge_db.get_queue_items(priority=0)
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_transport_error_is_transient_and_stops_drain(
        self, sync_engine, edge_db
    ) -> None:
        """Link failures must not burn dead-letter attempts — and once the
        first post fails, the rest of the FIFO drain is doomed too."""
        edge_db.enqueue_sync(2, {"data": "a"})
        edge_db.enqueue_sync(2, {"data": "b"})

        import httpx

        sync_engine._http_client.post = AsyncMock(
            side_effect=httpx.ConnectError("refused")
        )

        sent = await sync_engine.flush_queue(Priority.NORMAL)
        assert sent == 0
        assert sync_engine._http_client.post.call_count == 1  # broke after first

        items = edge_db.get_queue_items(priority=2)
        assert [i["attempts"] for i in items] == [0, 0]

    @pytest.mark.asyncio
    async def test_429_is_transient_and_stops_drain(self, sync_engine, edge_db) -> None:
        """Rate-limit backpressure must not consume the 10-attempt budget."""
        for i in range(5):
            edge_db.enqueue_sync(1, {"model": "battery_health", "n": i})

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        sync_engine._http_client.post = AsyncMock(return_value=mock_resp)

        sent = await sync_engine.flush_queue(Priority.HIGH)
        assert sent == 0
        assert sync_engine._http_client.post.call_count == 1

        items = edge_db.get_queue_items(priority=1)
        assert all(i["attempts"] == 0 for i in items)

    @pytest.mark.asyncio
    async def test_validation_422_burns_an_attempt(self, sync_engine, edge_db) -> None:
        edge_db.enqueue_sync(1, {"model": "battery_health"})

        mock_resp = MagicMock()
        mock_resp.status_code = 422
        sync_engine._http_client.post = AsyncMock(return_value=mock_resp)

        await sync_engine.flush_queue(Priority.HIGH)

        items = edge_db.get_queue_items(priority=1)
        assert items[0]["attempts"] == 1


class TestTelemetryDrain:
    @pytest.mark.asyncio
    async def test_drains_and_marks_synced(self, sync_engine, edge_db) -> None:
        now = int(time.time() * 1000)
        for i in range(3):
            edge_db.insert_measurement(now + i, ASSET_A, "battery.soc_pct", 70.0 + i, 0)

        mock_resp = MagicMock()
        mock_resp.status_code = 202
        sync_engine._http_client.post = AsyncMock(return_value=mock_resp)

        sent = await sync_engine.flush_telemetry()
        assert sent == 3
        assert edge_db.count_unsynced_measurements() == 0

        call = sync_engine._http_client.post.call_args
        assert call.args[0].endswith("/api/v1/ingest/telemetry")
        assert call.kwargs["headers"]["Content-Encoding"] == "gzip"
        assert call.kwargs["headers"]["X-Device-Id"] == "test-edge-001"
        body = json.loads(gzip.decompress(call.kwargs["content"]))
        assert len(body["measurements"]) == 3
        measurement = body["measurements"][0]
        assert measurement["metric_name"] == "battery.soc_pct"
        assert measurement["time"].endswith("+00:00")  # ISO 8601 UTC
        assert "site_id" not in measurement  # device registration is the authority

    @pytest.mark.asyncio
    async def test_quality_codes_map_to_cloud_scale(self, sync_engine, edge_db) -> None:
        """Edge 1=suspect/2=bad both map to cloud BAD (4) — questionable
        readings must never arrive as INTERPOLATED/ESTIMATED."""
        now = int(time.time() * 1000)
        edge_db.insert_measurement(now, ASSET_A, "battery.soc_pct", 50.0, 1)
        edge_db.insert_measurement(now + 1, ASSET_A, "battery.soc_pct", 51.0, 2)
        edge_db.insert_measurement(now + 2, ASSET_A, "battery.soc_pct", 52.0, 0)

        mock_resp = MagicMock()
        mock_resp.status_code = 202
        sync_engine._http_client.post = AsyncMock(return_value=mock_resp)

        await sync_engine.flush_telemetry()
        body = json.loads(
            gzip.decompress(sync_engine._http_client.post.call_args.kwargs["content"])
        )
        assert [m["quality"] for m in body["measurements"]] == [4, 4, 0]

    @pytest.mark.asyncio
    async def test_recent_pass_skips_backfill_rows(self, sync_engine, edge_db) -> None:
        now = int(time.time() * 1000)
        old = now - 48 * 3600_000  # outside the 24h recent window
        edge_db.insert_measurement(now, ASSET_A, "battery.soc_pct", 1.0, 0)
        edge_db.insert_measurement(old, ASSET_A, "battery.soc_pct", 2.0, 0)

        mock_resp = MagicMock()
        mock_resp.status_code = 202
        sync_engine._http_client.post = AsyncMock(return_value=mock_resp)

        assert await sync_engine.flush_telemetry(backfill=False) == 1
        assert edge_db.count_unsynced_measurements() == 1

        assert await sync_engine.flush_telemetry(backfill=True) == 1
        assert edge_db.count_unsynced_measurements() == 0

    @pytest.mark.asyncio
    async def test_transient_failure_leaves_rows_unsynced(
        self, sync_engine, edge_db
    ) -> None:
        import httpx

        now = int(time.time() * 1000)
        edge_db.insert_measurement(now, ASSET_A, "battery.soc_pct", 1.0, 0)
        sync_engine._http_client.post = AsyncMock(
            side_effect=httpx.ConnectError("down")
        )

        assert await sync_engine.flush_telemetry() == 0
        assert edge_db.count_unsynced_measurements() == 1

    @pytest.mark.asyncio
    async def test_auth_4xx_never_tombstones_rows(self, sync_engine, edge_db) -> None:
        """A 401 (device registered late) or 403 (temporarily blocked) is
        device-level, not row-level: rows must stay unsynced for retry —
        no bisect storm, no synced=2 tombstones."""
        now = int(time.time() * 1000)
        for i in range(4):
            edge_db.insert_measurement(now + i, ASSET_A, "battery.soc_pct", float(i), 0)

        for status in (401, 403, 404, 413):
            mock_resp = MagicMock()
            mock_resp.status_code = status
            sync_engine._http_client.post = AsyncMock(return_value=mock_resp)

            assert await sync_engine.flush_telemetry() == 0
            assert sync_engine._http_client.post.call_count == 1  # no bisect
            assert edge_db.count_unsynced_measurements() == 4
            assert edge_db.count_rejected_measurements() == 0

    @pytest.mark.asyncio
    async def test_validation_reject_isolates_poison_row(
        self, sync_engine, edge_db
    ) -> None:
        now = int(time.time() * 1000)
        edge_db.insert_measurement(now, ASSET_A, "battery.soc_pct", 1.0, 0)
        edge_db.insert_measurement(now + 1, ASSET_POISON, "bogus.metric", 2.0, 0)
        edge_db.insert_measurement(now + 2, ASSET_B, "battery.soc_pct", 3.0, 0)

        async def post(url, **kwargs):
            batch = json.loads(gzip.decompress(kwargs["content"]))
            has_poison = any(
                m["metric_name"] == "bogus.metric" for m in batch["measurements"]
            )
            resp = MagicMock()
            resp.status_code = 422 if has_poison else 202
            return resp

        sync_engine._http_client.post = AsyncMock(side_effect=post)

        handled = await sync_engine.flush_telemetry()
        assert handled == 3
        assert edge_db.count_unsynced_measurements() == 0

        # Poison row kept locally as rejected (2); good rows synced (1)
        states = dict(
            edge_db._conn.execute(
                "SELECT asset_id, synced FROM measurements"
            ).fetchall()
        )
        assert states[ASSET_POISON] == 2
        assert states[ASSET_A] == 1
        assert states[ASSET_B] == 1

    @pytest.mark.asyncio
    async def test_unmappable_rows_rejected_without_http(
        self, sync_engine, edge_db
    ) -> None:
        """Non-UUID asset ids (missing asset_mappings) and non-finite values
        can never validate cloud-side — mark rejected locally, zero POSTs,
        no 422 bisect storm."""
        now = int(time.time() * 1000)
        edge_db.insert_measurement(now, "vendor-raw-id-7", "battery.soc_pct", 1.0, 0)
        edge_db.insert_measurement(now + 1, ASSET_A, "battery.soc_pct", float("inf"), 0)

        sync_engine._http_client.post = AsyncMock()

        handled = await sync_engine.flush_telemetry()
        assert handled == 2
        assert sync_engine._http_client.post.call_count == 0
        assert edge_db.count_unsynced_measurements() == 0
        assert edge_db.count_rejected_measurements() == 2

    @pytest.mark.asyncio
    async def test_batch_size_and_cycle_cap(self, sync_config, edge_db) -> None:
        from dataclasses import replace

        config = replace(
            sync_config, telemetry_batch_size=2, telemetry_max_batches_per_cycle=2
        )
        engine = SyncEngine(config, edge_db)
        engine._http_client = AsyncMock()

        now = int(time.time() * 1000)
        for i in range(10):
            edge_db.insert_measurement(now + i, ASSET_A, "battery.soc_pct", float(i), 0)

        mock_resp = MagicMock()
        mock_resp.status_code = 202
        engine._http_client.post = AsyncMock(return_value=mock_resp)

        sent = await engine.flush_telemetry()
        assert sent == 4  # 2 batches × 2 rows, capped per cycle
        assert engine._http_client.post.call_count == 2
        assert edge_db.count_unsynced_measurements() == 6

    def test_batch_size_clamped_to_server_max(self, sync_config, edge_db) -> None:
        from dataclasses import replace

        config = replace(sync_config, telemetry_batch_size=50_000)
        engine = SyncEngine(config, edge_db)
        assert engine._telemetry_batch_size == 10_000


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_reads_watchdog_status(self, sync_engine, edge_db) -> None:
        import os
        import tempfile
        import time

        # Write a mock watchdog status file in a temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = os.path.join(tmpdir, "system.json")
            with open(status_path, "w") as f:
                json.dump(
                    {
                        "container_status": {"gen318-mqtt": "running"},
                        "system_metrics": {"cpu_pct": 25.0},
                        "updated_at": time.time(),
                    },
                    f,
                )

            edge_db.update_model_state("battery_health", "1.0.0", "a" * 64)

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            sync_engine._http_client.post = AsyncMock(return_value=mock_resp)

            # Patch the status file path
            with patch.object(
                sync_engine,
                "_read_watchdog_status",
                return_value={
                    "container_status": {"gen318-mqtt": "running"},
                    "system_metrics": {"cpu_pct": 25.0},
                },
            ):
                result = await sync_engine.send_heartbeat()
                # Returns the parsed body now (the cloud→device command
                # channel); None is the old `False`.
                assert result is not None

            call_kwargs = sync_engine._http_client.post.call_args
            body = call_kwargs.kwargs["json"]
            assert body["container_status"] == {"gen318-mqtt": "running"}
            assert body["system_metrics"]["cpu_pct"] == 25.0
            assert body["model_versions"]["battery_health"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_heartbeat_omits_cert_fingerprint_when_absent(
        self, sync_engine
    ) -> None:
        # The fixture config points at /tmp/fake.pem, which does not exist. An
        # old node, a dev laptop and CI all look like this — the key must not
        # appear at all, so the payload stays byte-identical to today's.
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        sync_engine._http_client.post = AsyncMock(return_value=mock_resp)

        assert sync_engine._device_cert_fingerprint is None
        assert await sync_engine.send_heartbeat() is not None
        assert (
            "device_cert_fingerprint"
            not in sync_engine._http_client.post.call_args.kwargs["json"]
        )

    @pytest.mark.asyncio
    async def test_heartbeat_reports_cert_fingerprint_when_present(
        self, sync_config, edge_db, tmp_path
    ) -> None:
        cert_path = tmp_path / "device.pem"
        cert_path.write_text(_SELF_SIGNED_PEM)
        sync_config.device_cert_path = str(cert_path)
        engine = SyncEngine(sync_config, edge_db)
        engine._http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        engine._http_client.post = AsyncMock(return_value=mock_resp)

        assert await engine.send_heartbeat() is not None
        body = engine._http_client.post.call_args.kwargs["json"]
        assert body["device_cert_fingerprint"] == _expected_fingerprint()

    @pytest.mark.asyncio
    async def test_fingerprint_is_computed_once_not_per_heartbeat(
        self, sync_config, edge_db, tmp_path
    ) -> None:
        # 60s heartbeats must not do file I/O; the certificate cannot change
        # under a running container anyway.
        cert_path = tmp_path / "device.pem"
        cert_path.write_text(_SELF_SIGNED_PEM)
        sync_config.device_cert_path = str(cert_path)
        engine = SyncEngine(sync_config, edge_db)
        engine._http_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        engine._http_client.post = AsyncMock(return_value=mock_resp)

        cert_path.unlink()
        await engine.send_heartbeat()
        body = engine._http_client.post.call_args.kwargs["json"]
        assert body["device_cert_fingerprint"] == _expected_fingerprint()


class TestLocalCertFingerprint:
    """The self-report that makes the mTLS cut-over pre-flightable.

    It must produce exactly what scripts/edge-pki.sh and Caddy produce, and it
    must degrade to None for every failure — a node that cannot read its own
    certificate still has to heartbeat, ingest and drain its backlog.
    """

    def test_matches_sha256_of_the_der_certificate(self, tmp_path) -> None:
        cert_path = tmp_path / "device.pem"
        cert_path.write_text(_SELF_SIGNED_PEM)
        assert _local_cert_fingerprint(str(cert_path)) == _expected_fingerprint()

    def test_tolerates_trailing_chain_and_leading_text(self, tmp_path) -> None:
        # openssl-issued device certs routinely carry a text dump above the PEM
        # block and the issuing CA below it; only the LEAF is fingerprinted.
        cert_path = tmp_path / "device.pem"
        cert_path.write_text(
            f"subject=CN=edge-001\n{_SELF_SIGNED_PEM}{_SELF_SIGNED_PEM}"
        )
        assert _local_cert_fingerprint(str(cert_path)) == _expected_fingerprint()

    def test_missing_file_returns_none(self, tmp_path) -> None:
        assert _local_cert_fingerprint(str(tmp_path / "nope.pem")) is None

    def test_directory_returns_none(self, tmp_path) -> None:
        assert _local_cert_fingerprint(str(tmp_path)) is None

    def test_non_pem_file_returns_none(self, tmp_path) -> None:
        junk = tmp_path / "device.pem"
        junk.write_text("this is not a certificate\n")
        assert _local_cert_fingerprint(str(junk)) is None

    def test_pem_without_an_end_marker_returns_none(self, tmp_path) -> None:
        broken = tmp_path / "device.pem"
        broken.write_text(_SELF_SIGNED_PEM.replace(_PEM_END_MARKER, ""))
        assert _local_cert_fingerprint(str(broken)) is None

    def test_undecodable_body_returns_none(self, tmp_path) -> None:
        broken = tmp_path / "device.pem"
        broken.write_text(f"{_PEM_BEGIN_MARKER}\nx\n{_PEM_END_MARKER}\n")
        assert _local_cert_fingerprint(str(broken)) is None

    def test_engine_construction_survives_an_absent_cert(
        self, sync_config, edge_db
    ) -> None:
        # The dev/test path: no certs, unverified client, loud warning. It must
        # not regress into an exception at construction time.
        engine = SyncEngine(sync_config, edge_db)
        assert engine._device_cert_fingerprint is None


class TestOTACheck:
    @pytest.mark.asyncio
    async def test_check_for_ota_detects_update(self, sync_engine, edge_db) -> None:
        edge_db.set_config("firmware_version", "1.0.0")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "update_available": True,
            "target_version": "2.0.0",
            "channel": "stable",
        }
        sync_engine._http_client.get = AsyncMock(return_value=mock_resp)

        mock_mqtt = MagicMock()
        result = await sync_engine.check_for_ota(mqtt_client=mock_mqtt)

        assert result is not None
        assert result["target_version"] == "2.0.0"
        mock_mqtt.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_for_ota_no_update(self, sync_engine, edge_db) -> None:
        edge_db.set_config("firmware_version", "2.0.0")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"update_available": False}
        sync_engine._http_client.get = AsyncMock(return_value=mock_resp)

        result = await sync_engine.check_for_ota()
        assert result is None
