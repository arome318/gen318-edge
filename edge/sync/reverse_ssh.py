"""On-demand reverse SSH tunnel for remote edge node access.

Two channels open a tunnel: the cloud's heartbeat-response command (the wired
path, applied on the asyncio loop by ``SyncService._handle_cloud_commands``)
and the node-local MQTT topic ``gen318/control/ssh`` (the on-site operator's
escape hatch).

Three properties are load-bearing and were absent before:

* **The cloud owns the deadline.** ``expires_at`` comes from
  ``edge_access_grants``. The old code clamped to a 30-minute class constant
  while the API allowed 60, so the ``auto_close_at`` the operator was shown
  could be half an hour later than the device's actual close.
* **A finished child never blocks a respawn.** ``is_open`` calls ``poll()``,
  which both detects and reaps an exited child, so ``open()`` on a dead handle
  starts a new tunnel instead of returning ``already_open`` forever. Within one
  process there is no orphan to hunt: the ssh child is ours.
* **A restart closes the books.** The watchdog restarts this container
  routinely, and everything in it dies with the container's PID namespace —
  including the ssh child and the in-process auto-close task. What survives is
  the cloud's belief that the tunnel is `active`. So the live grant (id, port,
  deadline and the report PROOF — never the token) is persisted to the SQLite
  ``config`` table, and ``SyncService`` reads it at startup and reports the
  grant `closed`. Cloud-side expiry still closes the record regardless; this
  makes the record match reality in seconds instead of at the deadline.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger()

# Key in the edge `config` table holding the live grant across restarts.
SSH_GRANT_CONFIG_KEY = "ssh_grant"


class ReverseSSHTunnel:
    """Manages reverse SSH tunnel from edge node to cloud jump host."""

    # gen318.com is the registered domain; this said ssh.gen318.io, which
    # resolves to nothing — the support tunnel could never have connected.
    # Env-overridable because hardcoding it is why it drifted unnoticed.
    SSH_SERVER = os.environ.get("EDGE_SSH_SERVER", "ssh.gen318.com")
    SSH_PORT = int(os.environ.get("EDGE_SSH_PORT", "22"))
    # Fallback only, for the local-MQTT path that supplies no cloud deadline.
    TIMEOUT_MINUTES = 30

    def __init__(self, device_id: str, device_key_path: str, db: Any = None) -> None:
        self.device_id = device_id
        self.device_key_path = device_key_path
        self.db = db
        self._process: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._opened_at: float | None = None
        self._close_task: asyncio.Task | None = None  # type: ignore[type-arg]

    @property
    def is_open(self) -> bool:
        return self._process is not None and self._process.poll() is None

    async def open(
        self,
        remote_port: int,
        timeout_minutes: int = 30,
        expires_at: str | None = None,
        grant_id: str | None = None,
        token_proof: str | None = None,
    ) -> dict:
        """Open reverse SSH tunnel. Remote port and deadline assigned by cloud.

        ``expires_at`` (ISO-8601, from the grant) wins over ``timeout_minutes``
        so the cloud is the single source of the deadline.

        ``token_proof`` is persisted with the grant so a container restarted by
        the watchdog can still report the grant closed. The token itself is
        never written down here or anywhere else.
        """
        if self.is_open:
            return {
                "status": "already_open",
                "remote_port": remote_port,
                "device_id": self.device_id,
            }
        # A handle we still hold whose child has exited: is_open already called
        # poll(), which reaped it, so simply spawn again. Dropping the stale
        # handle keeps `_process` meaning "the child we believe is running".
        self._process = None

        self._process = subprocess.Popen(
            [
                "ssh",
                "-N",
                "-R",
                f"{remote_port}:localhost:22",
                f"tunnel@{self.SSH_SERVER}",
                "-p",
                str(self.SSH_PORT),
                "-i",
                self.device_key_path,
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                "UserKnownHostsFile=/etc/gen318/known_hosts",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._opened_at = time.time()

        deadline = _parse_deadline(expires_at)
        if deadline is None:
            # Local-MQTT path: no cloud grant, so fall back to the class cap.
            deadline = self._opened_at + min(timeout_minutes, self.TIMEOUT_MINUTES) * 60
        delay = max(0.0, deadline - self._opened_at)

        # Schedule auto-close. Belt-and-braces only: the cloud sweep is what
        # makes the deadline authoritative, because this task does not survive
        # a watchdog container restart.
        loop = asyncio.get_running_loop()
        self._close_task = loop.create_task(self._auto_close(delay))

        auto_close_at = datetime.fromtimestamp(deadline, tz=UTC)
        self._persist_grant(grant_id, remote_port, auto_close_at, token_proof)

        logger.info(
            "ssh_tunnel_opened",
            device_id=self.device_id,
            grant_id=grant_id,
            remote_port=remote_port,
            auto_close_at=auto_close_at.isoformat(),
        )

        return {
            "status": "opened",
            "remote_port": remote_port,
            "device_id": self.device_id,
            "grant_id": grant_id,
            "auto_close_at": auto_close_at.isoformat(),
        }

    def _persist_grant(
        self,
        grant_id: str | None,
        remote_port: int,
        auto_close_at: datetime,
        token_proof: str | None = None,
    ) -> None:
        """Record the live grant so a restarted container can close the books.

        Read at startup by ``SyncService._recover_orphaned_grant``. Uses the
        existing SQLite `config` table — no edge schema change — and never
        raises: losing the record costs visibility, not the tunnel.

        The stored ``token_proof`` is a one-way value bound to this grant id. It
        cannot open anything; it is only what the cloud needs to accept the
        `closed` report. The token itself is never persisted.
        """
        if self.db is None:
            return
        try:
            self.db.set_config(
                SSH_GRANT_CONFIG_KEY,
                json.dumps(
                    {
                        "grant_id": grant_id,
                        "remote_port": remote_port,
                        "opened_at": self._opened_at,
                        "expires_at": auto_close_at.isoformat(),
                        "token_proof": token_proof,
                    }
                ),
            )
        except Exception:
            logger.warning("ssh_grant_persist_failed", grant_id=grant_id)

    def _clear_grant(self) -> None:
        if self.db is None:
            return
        try:
            self.db.set_config(SSH_GRANT_CONFIG_KEY, "")
        except Exception:
            logger.warning("ssh_grant_clear_failed")

    async def _auto_close(self, delay_seconds: float) -> None:
        """Auto-close tunnel after timeout."""
        await asyncio.sleep(delay_seconds)
        await self.close()
        logger.info(
            "ssh_tunnel_auto_closed",
            device_id=self.device_id,
            reason="timeout",
        )

    async def close(self) -> dict:
        """Close the reverse SSH tunnel."""
        duration = 0
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()

        if self._opened_at:
            duration = int(time.time() - self._opened_at)

        if self._close_task and not self._close_task.done():
            self._close_task.cancel()

        self._process = None
        self._opened_at = None
        self._close_task = None
        self._clear_grant()

        logger.info(
            "ssh_tunnel_closed",
            device_id=self.device_id,
            duration_seconds=duration,
        )

        return {
            "status": "closed",
            "device_id": self.device_id,
            "duration_seconds": duration,
        }

    def get_status(self) -> dict:
        """Get current tunnel status."""
        if self.is_open and self._opened_at:
            elapsed = int(time.time() - self._opened_at)
            remaining = max(0, self.TIMEOUT_MINUTES * 60 - elapsed)
            return {
                "status": "open",
                "elapsed_seconds": elapsed,
                "remaining_seconds": remaining,
            }
        return {"status": "closed"}


def _parse_deadline(expires_at: str | None) -> float | None:
    """ISO-8601 → unix seconds. Returns None on anything unparseable so the
    caller falls back to its own cap rather than opening an unbounded tunnel."""
    if not expires_at:
        return None
    try:
        parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("ssh_grant_deadline_unparseable", expires_at=expires_at)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()
