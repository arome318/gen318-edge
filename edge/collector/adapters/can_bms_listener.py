"""Receive-only CAN listener for Victron-profile BMS packs.

The Pi taps an existing inverter-to-battery CAN bus and reads what is already
being said on it. It never transmits. Two active masters on a BMS bus is how
a battery ends up disconnecting itself, and this adapter is the only thing
standing between a monitoring install and that outcome, so the rule is
enforced in three independent places:

1. **There is no transmit path in this class.** No ``send``, no
   ``send_periodic``, no notifier with a writer. The required ``write()`` from
   the adapter contract exists only to refuse.
2. **The bus is opened with ``receive_own_messages=False``** — necessary, and
   nowhere near sufficient on its own.
3. **The interface must ALREADY be in listen-only mode**, which is checked by
   reading ``ip -details link show <interface>`` before connecting. If
   listen-only cannot be CONFIRMED, the adapter refuses to start.

Point 3 is deliberately a check and never a fix. This adapter does not run
``ip link set`` — reconfiguring a shared interface would disturb whatever
else is on that bus, and a monitoring tool that silently reconfigures network
hardware is worse than one that refuses to start. Setting listen-only belongs
to the container entrypoint or the commissioning procedure, both of which are
documented in docs/ONBOARDING_DEYE_HYBRID.md.

Offline-first: python-can absent, interface missing, or the bus unreadable
all degrade to a logged refusal. Nothing here can block telemetry collection
from any other adapter.
"""

from __future__ import annotations

import asyncio
import contextlib
import struct
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import structlog

from .base import AdapterCapabilities, AdapterStatus, HardwareAdapter, Reading
from .victron_can_profile import (
    ALARM_FRAME_ID,
    PROFILES,
    decode_alarm,
    decode_fields,
    decode_identity,
    supported_metrics,
)

logger = structlog.get_logger()

#: Bound on frames drained per poll. A CAN bus can produce thousands per
#: second; the collector runs under a 256 MB limit and a poll that tries to
#: drain everything would be a memory bomb rather than a reading.
#: Returned by the interface probe when the ``ip`` binary is absent, which is
#: a different problem from an interface that is simply not listen-only.
IP_NOT_FOUND = "__ip_not_found__"

_MAX_FRAMES_PER_POLL = 400
_POLL_WINDOW_SECONDS = 2.0
_FRAME_TIMEOUT_SECONDS = 0.5


class CANBMSListenerAdapter(HardwareAdapter):
    """Reads a Victron-profile BMS bus. Never writes to it."""

    def __init__(self) -> None:
        self._bus: Any = None
        self._notifier: Any = None
        self._reader: Any = None
        self._config: dict[str, Any] = {}
        self._status = AdapterStatus.DISCONNECTED
        self._last_reading_at: float = 0.0
        self._asset_id: str = ""
        self._profile: str = "victron_generic"
        self._interface: str = "can0"
        self._identity: dict[str, str] = {}

    # ── Listen-only verification ─────────────────────────────────

    @staticmethod
    async def _interface_state(interface: str) -> str | None:
        """Raw ``ip -details link show`` output, or None if it cannot be read.

        None means "could not confirm", which this adapter treats exactly like
        "not listen-only": it refuses. A container without permission to query
        the interface is not evidence of safety.

        ``IP_NOT_FOUND`` is a NARROWER kind of None: the ``ip`` binary is not
        in this image at all. Behaviour is identical — still a refusal — but
        the two produce different log lines, because they send a technician to
        completely different places. Without the distinction, a node missing
        iproute2 tells the person to run the exact command they already ran on
        the host, and they will run it again, restart, and read the same line.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "ip",
                "-details",
                "link",
                "show",
                interface,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return IP_NOT_FOUND
        except (OSError, ValueError):
            return None
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5.0)
        except asyncio.TimeoutError:
            # Do not leave the child behind on a busy or wedged node.
            with contextlib.suppress(ProcessLookupError, OSError):
                process.kill()
            return None
        except (OSError, ValueError):
            return None
        if process.returncode != 0:
            return None
        return stdout.decode("utf-8", errors="replace")

    @classmethod
    def _is_listen_only(cls, state: str | None) -> bool:
        """True only when the interface reports listen-only.

        iproute2 prints the flag as ``LISTEN-ONLY`` among the CAN control
        mode flags. Absence of the string is treated as absence of the mode,
        and so is every "could not ask" outcome.
        """
        if not state or state == IP_NOT_FOUND:
            return False
        return "listen-only" in state.lower()

    async def _refuse_unless_listen_only(self) -> bool:
        if self._config.get("require_listen_only", True) is False:
            # A deliberate, explicit opt-out for a bench rig on an isolated
            # bus. It has to be written down in the site config to happen.
            logger.warning(
                "can_listener_listen_only_check_disabled",
                interface=self._interface,
                msg="require_listen_only is false — only safe on an isolated bus",
            )
            return True
        state = await self._interface_state(self._interface)
        if self._is_listen_only(state):
            return True
        if state == IP_NOT_FOUND:
            # Same refusal, different remedy. Telling the technician to set
            # listen-only here would be wrong: they very likely already did,
            # on the host, and this container simply cannot see it.
            logger.error(
                "can_listener_listen_only_unverifiable",
                interface=self._interface,
                reason="iproute2_missing",
                msg=(
                    "the 'ip' command is not present in this image, so "
                    "listen-only cannot be verified and the bus stays closed. "
                    "This is a build problem, not a wiring one: install "
                    "iproute2 in the collector image. Re-running 'ip link set' "
                    "on the host will not change this message."
                ),
            )
            return False
        logger.error(
            "can_listener_refusing_not_listen_only",
            interface=self._interface,
            confirmed=state is not None,
            msg=(
                "the interface is not confirmed listen-only; refusing to open "
                "the bus. Set it before starting the collector, e.g. "
                f"'ip link set {self._interface} type can bitrate 500000 "
                "listen-only on'. This adapter never reconfigures an interface "
                "it shares with an inverter and a battery."
            ),
        )
        return False

    # ── Lifecycle ────────────────────────────────────────────────

    async def initialize(self, config: dict[str, Any]) -> bool:
        self._config = config
        self._asset_id = str(config.get("asset_id", ""))
        self._interface = str(config.get("interface", "can0"))
        profile = str(config.get("profile", "victron_generic"))
        if profile not in PROFILES:
            logger.error(
                "can_listener_unknown_profile",
                profile=profile,
                available=list(PROFILES),
            )
            self._status = AdapterStatus.ERROR
            return False
        self._profile = profile
        if not self._asset_id:
            logger.error("can_listener_missing_asset_id")
            self._status = AdapterStatus.ERROR
            return False

        if not await self._refuse_unless_listen_only():
            self._status = AdapterStatus.ERROR
            return False

        try:
            import can

            self._bus = can.Bus(
                interface="socketcan",
                channel=self._interface,
                bitrate=int(config.get("bitrate", 500000)),
                # Necessary but NOT sufficient — see the listen-only check
                # above, which is the control that actually holds.
                receive_own_messages=False,
            )
            self._reader = can.AsyncBufferedReader()
            loop = asyncio.get_running_loop()
            # A notifier with listeners only. No writer is ever attached.
            self._notifier = can.Notifier(self._bus, [self._reader], loop=loop)
        except Exception:
            logger.exception("can_listener_init_failed", interface=self._interface)
            self._status = AdapterStatus.ERROR
            return False

        self._status = AdapterStatus.CONNECTED
        logger.info(
            "can_listener_initialized",
            interface=self._interface,
            profile=self._profile,
            bitrate=config.get("bitrate", 500000),
            asset_id=self._asset_id,
        )
        return True

    async def poll(self) -> list[Reading]:
        """Drain what the bus has said, bounded in both frames and time."""
        if not self._reader:
            return []

        readings: list[Reading] = []
        now = datetime.now(UTC)
        deadline = time.time() + _POLL_WINDOW_SECONDS
        frames = 0

        while time.time() < deadline and frames < _MAX_FRAMES_PER_POLL:
            try:
                message = await asyncio.wait_for(
                    self._reader.get_message(), timeout=_FRAME_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                break
            if message is None:
                continue
            frames += 1
            try:
                readings.extend(self._parse(message, now))
            except (struct.error, IndexError, TypeError, ValueError) as exc:
                # A flaky transceiver produces frames with no payload or a
                # payload that is not bytes at all. One bad frame must cost
                # that frame and nothing else — the drain keeps going, and
                # the poll still returns whatever else the bus said.
                logger.warning(
                    "can_listener_parse_error",
                    arb_id=f"0x{getattr(message, 'arbitration_id', 0):03X}",
                    error=str(exc),
                )

        if readings:
            self._last_reading_at = time.time()
        return readings

    def _parse(self, message: Any, now: datetime) -> list[Reading]:
        arb_id: int = message.arbitration_id
        data: bytes = bytes(message.data)

        if arb_id == ALARM_FRAME_ID:
            return decode_alarm(data, self._asset_id, now)

        identity = decode_identity(arb_id, data)
        if identity is not None:
            self._identity[identity[0]] = identity[1]
            return []

        return decode_fields(arb_id, data, self._asset_id, now)

    def identity(self) -> dict[str, str]:
        """Manufacturer and model strings seen on the bus. Never telemetry."""
        return dict(self._identity)

    async def subscribe(
        self,
        callback: Callable[[list[Reading]], Awaitable[None]],
    ) -> bool:
        """CAN is push-based; this wraps the bounded drain in a loop."""
        while True:
            readings = await self.poll()
            if readings:
                await callback(readings)
            await asyncio.sleep(1)

    async def write(self, metric_name: str, value: float) -> bool:
        """Always False. This adapter has no transmit path at all.

        The method exists because the adapter contract requires it. Two active
        masters on a BMS bus is how a battery disconnects itself.
        """
        logger.warning("can_listener_write_refused", metric=metric_name)
        return False

    async def health_check(self) -> AdapterStatus:
        if not self._bus:
            return AdapterStatus.DISCONNECTED
        if self._last_reading_at > 0 and time.time() - self._last_reading_at > 120:
            return AdapterStatus.ERROR
        return AdapterStatus.CONNECTED

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            can_read=True,
            can_write=False,
            can_subscribe=True,
            supported_metrics=supported_metrics(),
            supported_write_metrics=[],
            poll_interval_seconds=1,
            min_poll_interval_seconds=1,
        )

    async def shutdown(self) -> None:
        if self._notifier:
            self._notifier.stop()
        if self._bus:
            self._bus.shutdown()
        self._status = AdapterStatus.DISCONNECTED
        logger.info("can_listener_shutdown", asset_id=self._asset_id)
