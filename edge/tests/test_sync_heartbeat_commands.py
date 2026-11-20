"""The cloud→device command channel: heartbeat response → local action.

This is the only place the cloud can ask an edge node to do anything, and it
is a RESPONSE, not a push — the node stays strictly pull-only. The properties
worth pinning are the offline ones, because they are the ones a refactor
quietly breaks:

* an offline heartbeat applies nothing and behaves exactly as it did before
  the response body was ever read;
* a command handler that raises must not kill the sync cycle, or one bad
  command takes the telemetry drain down with it;
* the grant outcome is enqueued to the P1 queue BEFORE any network call, so a
  link that drops mid-command still reports the truth on the next drain;
* the `ota_check` hint only triggers an off-cycle poll on ≥4G, matching the
  scheduled path — the tier ladder is contract.

Every async test needs an explicit @pytest.mark.asyncio: there is no
asyncio_mode under edge/.
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from sync.engine import ConnectivityState, Priority, SyncConfig, SyncEngine


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
def engine(sync_config, edge_db):
    eng = SyncEngine(sync_config, edge_db)
    eng._http_client = AsyncMock()
    return eng


def _response(body: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    return resp


class TestHeartbeatResponse:
    @pytest.mark.asyncio
    async def test_returns_parsed_body(self, engine) -> None:
        engine._http_client.post = AsyncMock(
            return_value=_response({"status": "heartbeat_accepted", "commands": []})
        )
        assert await engine.send_heartbeat() == {
            "status": "heartbeat_accepted",
            "commands": [],
        }

    @pytest.mark.asyncio
    async def test_offline_returns_none_and_applies_nothing(self, engine) -> None:
        # Identical to today's `return False` path — the offline branch of the
        # sync cycle must not change shape because the body is now read.
        engine._http_client.post = AsyncMock(side_effect=httpx.ConnectError("down"))
        handler = AsyncMock()
        engine._on_commands = handler

        result = await engine.send_heartbeat()
        await engine._apply_commands(result)

        assert result is None
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_2xx_returns_none(self, engine) -> None:
        engine._http_client.post = AsyncMock(return_value=_response({}, status=503))
        assert await engine.send_heartbeat() is None

    @pytest.mark.asyncio
    async def test_non_json_body_is_an_empty_dict_not_a_failure(self, engine) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        engine._http_client.post = AsyncMock(return_value=resp)
        assert await engine.send_heartbeat() == {}

    @pytest.mark.asyncio
    async def test_reports_the_running_firmware_version(self, engine, edge_db) -> None:
        edge_db.set_config("firmware_version", "1.4.2")
        engine._http_client.post = AsyncMock(return_value=_response({}))

        await engine.send_heartbeat()

        body = engine._http_client.post.call_args.kwargs["json"]
        assert body["firmware_version"] == "1.4.2"

    @pytest.mark.asyncio
    async def test_firmware_version_omitted_when_never_stamped(self, engine) -> None:
        engine._http_client.post = AsyncMock(return_value=_response({}))
        await engine.send_heartbeat()
        assert (
            "firmware_version" not in engine._http_client.post.call_args.kwargs["json"]
        )

    @pytest.mark.asyncio
    async def test_storage_runway_is_included_in_system_metrics(
        self, engine, edge_db
    ) -> None:
        edge_db.set_config(
            "storage.status",
            json.dumps(
                {
                    "pressure": "warning",
                    "estimated_runway_days": 18.5,
                    "target_days": 30,
                    "target_met": False,
                    "database_bytes": 1234,
                }
            ),
        )
        engine._http_client.post = AsyncMock(return_value=_response({}))

        await engine.send_heartbeat()

        metrics = engine._http_client.post.call_args.kwargs["json"]["system_metrics"]
        assert metrics["storage_pressure"] == "warning"
        assert metrics["storage_runway_days"] == 18.5
        assert metrics["storage_target_met"] is False


class TestApplyCommands:
    @pytest.mark.asyncio
    async def test_commands_reach_the_handler(self, engine) -> None:
        handler = AsyncMock()
        engine._on_commands = handler
        commands = [{"type": "open_tunnel", "grant_id": "g1", "remote_port": 22222}]

        await engine._apply_commands({"commands": commands})

        handler.assert_awaited_once_with(commands)

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_propagate(self, engine) -> None:
        # A command failure must never break a sync cycle: the cycle also
        # carries the telemetry drain and the retention sweep.
        engine._on_commands = AsyncMock(side_effect=RuntimeError("ssh exploded"))
        await engine._apply_commands(
            {"commands": [{"type": "close_tunnel", "grant_id": "g"}]}
        )

    @pytest.mark.asyncio
    async def test_no_handler_wired_is_a_no_op(self, engine) -> None:
        engine._on_commands = None
        await engine._apply_commands({"commands": [{"type": "ota_check"}]})

    @pytest.mark.asyncio
    async def test_malformed_commands_are_ignored(self, engine) -> None:
        handler = AsyncMock()
        engine._on_commands = handler
        await engine._apply_commands({"commands": "not-a-list"})
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ota_check_hint_sets_the_off_cycle_flag(self, engine) -> None:
        engine._on_commands = AsyncMock()
        await engine._apply_commands({"commands": [{"type": "ota_check"}]})
        assert engine._ota_check_requested is True


class TestOtaCheckHint:
    @pytest.mark.asyncio
    async def test_hint_triggers_an_off_cycle_poll_on_4g(self, engine) -> None:
        engine.check_connectivity = AsyncMock(return_value=ConnectivityState.FOURG)
        engine.send_heartbeat = AsyncMock(
            return_value={"commands": [{"type": "ota_check"}]}
        )
        engine.flush_queue = AsyncMock(return_value=0)
        engine.flush_telemetry = AsyncMock(return_value=0)
        engine.check_for_ota = AsyncMock(return_value=None)
        engine._on_commands = AsyncMock()

        # Cycle 1 is nowhere near OTA_CHECK_INTERVAL_CYCLES (360).
        await engine._sync_cycle(1)

        engine.check_for_ota.assert_awaited_once()
        # Consumed, so it does not poll every cycle thereafter.
        assert engine._ota_check_requested is False

    @pytest.mark.asyncio
    async def test_hint_does_not_poll_on_3g(self, engine) -> None:
        # The tier ladder is contract: OTA polling is a ≥4G activity, hint or
        # not. A 3G link is for the queue and recent telemetry.
        engine.check_connectivity = AsyncMock(return_value=ConnectivityState.THREEG)
        engine.send_heartbeat = AsyncMock(
            return_value={"commands": [{"type": "ota_check"}]}
        )
        engine.flush_queue = AsyncMock(return_value=0)
        engine.flush_telemetry = AsyncMock(return_value=0)
        engine.check_for_ota = AsyncMock(return_value=None)
        engine._on_commands = AsyncMock()

        await engine._sync_cycle(1)

        engine.check_for_ota.assert_not_awaited()
        assert engine._ota_check_requested is True

    @pytest.mark.asyncio
    async def test_2g_sends_no_heartbeat_so_no_commands_deliver(self, engine) -> None:
        # A documented consequence of the tier contract, not a bug: grants and
        # hints simply do not reach a 2G site.
        engine.check_connectivity = AsyncMock(return_value=ConnectivityState.TWOG)
        engine.send_heartbeat = AsyncMock()
        engine.flush_queue = AsyncMock(return_value=0)
        engine._on_commands = AsyncMock()

        await engine._sync_cycle(1)

        engine.send_heartbeat.assert_not_awaited()
        engine._on_commands.assert_not_awaited()


def _sync_service(edge_config, edge_db):
    from sync.main import SyncService

    svc = SyncService.__new__(SyncService)
    svc.config = edge_config
    svc.db = edge_db
    svc.mqtt = MagicMock()
    svc.engine = MagicMock()
    svc.ssh_tunnel = AsyncMock()
    svc._loop = None
    return svc


class TestGrantHandling:
    def _service(self, edge_config, edge_db):
        return _sync_service(edge_config, edge_db)

    @pytest.mark.asyncio
    async def test_open_tunnel_reports_outcome_on_the_p1_queue(
        self, edge_config, edge_db
    ) -> None:
        svc = self._service(edge_config, edge_db)
        svc.ssh_tunnel.open = AsyncMock(return_value={"status": "opened"})

        await svc._handle_cloud_commands(
            [
                {
                    "type": "open_tunnel",
                    "grant_id": "11111111-1111-1111-1111-111111111111",
                    "remote_port": 22222,
                    "expires_at": "2030-01-01T00:00:00+00:00",
                    "token": "tok",
                }
            ]
        )

        items = edge_db.get_queue_items(priority=Priority.HIGH)
        assert len(items) == 1
        payload = items[0]["payload"]
        assert payload["type"] == "edge.access_grant"
        assert payload["status"] == "opened"
        # A PROOF of the token, never the token. Everything queued here is
        # persisted verbatim by the cloud's critical events_raw_handler into
        # event_outbox.payload and events.payload, so echoing the token would
        # leave a plaintext shell credential in two tables for the life of
        # those rows.
        assert "token" not in payload
        assert (
            payload["token_proof"]
            == hashlib.sha256(b"tok:11111111-1111-1111-1111-111111111111").hexdigest()
        )

    @pytest.mark.asyncio
    async def test_the_proof_is_persisted_not_the_token(
        self, edge_config, edge_db
    ) -> None:
        # The restart-recovery record has to be enough to close the grant, and
        # must still never contain the credential itself.
        svc = self._service(edge_config, edge_db)
        captured: dict = {}

        async def _open(**kwargs):
            captured.update(kwargs)
            return {"status": "opened"}

        svc.ssh_tunnel.open = _open
        await svc._handle_cloud_commands(
            [
                {
                    "type": "open_tunnel",
                    "grant_id": "g-1",
                    "remote_port": 22222,
                    "token": "tok",
                }
            ]
        )
        assert captured["token_proof"] == hashlib.sha256(b"tok:g-1").hexdigest()
        assert "tok" not in str(list(captured.values()))

    @pytest.mark.asyncio
    async def test_open_failure_is_reported_not_swallowed(
        self, edge_config, edge_db
    ) -> None:
        svc = self._service(edge_config, edge_db)
        svc.ssh_tunnel.open = AsyncMock(side_effect=OSError("no ssh binary"))

        await svc._handle_cloud_commands(
            [{"type": "open_tunnel", "grant_id": "g", "remote_port": 22222}]
        )

        items = edge_db.get_queue_items(priority=Priority.HIGH)
        assert [i["payload"]["status"] for i in items] == ["failed"]

    @pytest.mark.asyncio
    async def test_close_tunnel_closes_and_reports(self, edge_config, edge_db) -> None:
        svc = self._service(edge_config, edge_db)
        svc.ssh_tunnel.close = AsyncMock(return_value={"status": "closed"})

        await svc._handle_cloud_commands([{"type": "close_tunnel", "grant_id": "g"}])

        svc.ssh_tunnel.close.assert_awaited_once()
        items = edge_db.get_queue_items(priority=Priority.HIGH)
        assert [i["payload"]["status"] for i in items] == ["closed"]

    @pytest.mark.asyncio
    async def test_unknown_command_is_ignored(self, edge_config, edge_db) -> None:
        svc = self._service(edge_config, edge_db)
        await svc._handle_cloud_commands([{"type": "self_destruct"}])
        assert edge_db.get_queue_items(priority=Priority.HIGH) == []


class TestReverseSSHDeadline:
    def _tunnel(self, edge_db):
        from sync.reverse_ssh import ReverseSSHTunnel

        return ReverseSSHTunnel("test-edge-001", "/tmp/fake.key", db=edge_db)

    def test_parses_the_cloud_deadline(self) -> None:
        from sync.reverse_ssh import _parse_deadline

        assert _parse_deadline("2030-01-01T00:00:00Z") == pytest.approx(
            1893456000.0, abs=1.0
        )

    def test_unparseable_deadline_falls_back_rather_than_opening_unbounded(
        self,
    ) -> None:
        from sync.reverse_ssh import _parse_deadline

        assert _parse_deadline("whenever") is None
        assert _parse_deadline(None) is None

    @pytest.mark.asyncio
    async def test_open_persists_the_grant_and_close_clears_it(
        self, edge_db, monkeypatch
    ) -> None:
        # The watchdog restarts containers routinely and the auto-close timer
        # is an in-process task; the persisted record is what lets a restarted
        # container close the books instead of leaving the cloud believing a
        # dead tunnel is live.
        import json

        from sync.reverse_ssh import SSH_GRANT_CONFIG_KEY

        tunnel = self._tunnel(edge_db)
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        monkeypatch.setattr(
            "sync.reverse_ssh.subprocess.Popen", lambda *a, **k: fake_proc
        )

        await tunnel.open(
            22222,
            expires_at="2030-01-01T00:00:00Z",
            grant_id="g-1",
            token_proof="proof-value",
        )
        stored = json.loads(edge_db.get_config(SSH_GRANT_CONFIG_KEY))
        assert stored["grant_id"] == "g-1"
        assert stored["remote_port"] == 22222
        # Enough to report the grant closed, and useless for opening anything.
        assert stored["token_proof"] == "proof-value"

        await tunnel.close()
        assert edge_db.get_config(SSH_GRANT_CONFIG_KEY) == ""

    @pytest.mark.asyncio
    async def test_dead_child_does_not_block_respawn(
        self, edge_db, monkeypatch
    ) -> None:
        # is_open() calls poll(), which both detects and reaps an exited child,
        # so a stale handle must not make open() return 'already_open' forever
        # while nothing is listening on the remote port.
        tunnel = self._tunnel(edge_db)
        dead = MagicMock()
        dead.poll.return_value = 0  # already exited
        tunnel._process = dead

        live = MagicMock()
        live.poll.return_value = None
        monkeypatch.setattr("sync.reverse_ssh.subprocess.Popen", lambda *a, **k: live)

        result = await tunnel.open(22222, expires_at="2030-01-01T00:00:00Z")

        assert result["status"] == "opened"
        assert tunnel._process is live


class TestRestartRecovery:
    """A restarted container must close the books on the grant it lost.

    Everything in this container dies with its PID namespace — the ssh child
    and the in-process auto-close task both go. What survives is the cloud's
    belief that the tunnel is `active`, and a remote-shell audit record that
    disagrees with the hardware is the one thing it must never be. The
    persisted grant used to be written by ``_persist_grant`` and read by
    nothing, so the record stayed open until the cloud's expiry sweep.
    """

    def _stored(self, edge_db, **overrides) -> None:
        import json

        from sync.reverse_ssh import SSH_GRANT_CONFIG_KEY

        record = {
            "grant_id": "g-9",
            "remote_port": 22222,
            "opened_at": 1.0,
            "expires_at": "2030-01-01T00:00:00+00:00",
            "token_proof": "proof-value",
        }
        record.update(overrides)
        edge_db.set_config(SSH_GRANT_CONFIG_KEY, json.dumps(record))

    def test_an_orphaned_grant_is_reported_closed(self, edge_config, edge_db) -> None:
        from sync.reverse_ssh import SSH_GRANT_CONFIG_KEY

        svc = _sync_service(edge_config, edge_db)
        self._stored(edge_db)

        svc._recover_orphaned_grant()

        items = edge_db.get_queue_items(priority=Priority.HIGH)
        assert [
            (
                i["payload"]["grant_id"],
                i["payload"]["status"],
                i["payload"]["token_proof"],
            )
            for i in items
        ] == [("g-9", "closed", "proof-value")]
        # Cleared, so a second restart does not report it again.
        assert edge_db.get_config(SSH_GRANT_CONFIG_KEY) == ""

    def test_no_persisted_grant_is_a_no_op(self, edge_config, edge_db) -> None:
        svc = _sync_service(edge_config, edge_db)
        svc._recover_orphaned_grant()
        assert edge_db.get_queue_items(priority=Priority.HIGH) == []

    def test_a_corrupt_record_is_cleared_without_a_report(
        self, edge_config, edge_db
    ) -> None:
        from sync.reverse_ssh import SSH_GRANT_CONFIG_KEY

        svc = _sync_service(edge_config, edge_db)
        edge_db.set_config(SSH_GRANT_CONFIG_KEY, "{not json")

        svc._recover_orphaned_grant()

        assert edge_db.get_queue_items(priority=Priority.HIGH) == []
        assert edge_db.get_config(SSH_GRANT_CONFIG_KEY) == ""

    def test_recovery_never_stops_the_service_starting(
        self, edge_config, edge_db, monkeypatch
    ) -> None:
        # Offline-first rule 1: nothing on the startup path may fail hard.
        svc = _sync_service(edge_config, edge_db)
        self._stored(edge_db)
        monkeypatch.setattr(
            edge_db, "enqueue_sync", MagicMock(side_effect=RuntimeError("disk full"))
        )
        svc._recover_orphaned_grant()

    def test_a_stubbed_database_is_survivable(self, edge_config) -> None:
        # A MagicMock db returns a MagicMock from get_config, which is truthy
        # but not JSON. Startup must shrug, not crash.
        svc = _sync_service(edge_config, MagicMock())
        svc._recover_orphaned_grant()
