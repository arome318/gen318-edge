"""Voltronic-core inverter adapter — Axpert and its many rebadges.

Speaks the ASCII command protocol in ``voltronic_protocol``: PI30 on older
units, PI18 on newer ones, over 2400-baud serial or a USB HID interface. See
that module for the protocol, its sources, and why the command set is an
allowlist rather than a denylist.

Offline-first, like every adapter here. It reaches nothing but the local
serial port, a failed exchange returns no readings rather than raising, and
losing the inverter mid-session drops that poll and retries on the next one.

READ-ONLY. ``write()`` always returns False and logs. The protocol's setter
commands are never framed, because ``voltronic_protocol`` refuses to frame
anything outside the read allowlist.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import structlog

from .base import AdapterCapabilities, AdapterStatus, HardwareAdapter, Reading
from .voltronic_protocol import (
    SUPPORTED_METRICS,
    Reply,
    VoltronicProtocolError,
    frame_pi18,
    frame_pi30,
    parse_pi18_gs,
    parse_pi18_mod,
    parse_pi18_reply,
    parse_pi30_reply,
    parse_qmod,
    parse_qpigs,
    parse_qpiws,
)

logger = structlog.get_logger()

#: The inverter is slow and half-duplex. V1 sleeps 0.25 s after sending
#: before it starts reading, and paces HID chunk writes at 0.05 s.
_SETTLE_SECONDS = 0.25
_CHUNK_PAUSE_SECONDS = 0.05
_HID_REPORT_BYTES = 8
_READ_CHUNK = 256
_TERMINATOR = b"\r"

#: Voltronic's own documented line settings. The baud rate is not negotiable
#: on these units — 2400 is what the vendor document specifies and what both
#: reference implementations hard-code.
DEFAULT_BAUDRATE = 2400


#: How long ``close()`` waits for an in-flight worker thread to notice that
#: it should stop. Slightly longer than the default exchange timeout, so a
#: worker that is mid-read gets to finish rather than being abandoned.
_JOIN_TIMEOUT_SECONDS = 8.0


class _Transport:
    """One request, one reply, over whichever link the unit exposes.

    THE LIFETIME RULE, which is the whole reason this base class exists.

    Both transports do their I/O on a worker thread, because pyserial and
    ``os.read`` are synchronous and a blocking read on the event loop would
    stall every other adapter for the length of the inverter's timeout.

    But cancelling the ``await`` does not stop the thread. The collector can
    cancel a poll and immediately reconnect the adapter, and nothing in the
    supervisor joins the worker — so without care, a thread that was mid-read
    keeps issuing syscalls on a descriptor that has already been closed and,
    on Linux, reissued to the very next ``open``. That is not a lost reading:
    it is Voltronic command bytes written into whatever now holds that number,
    which on this node could be the SQLite handle.

    So every transport here obeys three rules:

      1. The worker re-reads the handle from ``self`` before each syscall and
         bails the moment it is gone, rather than capturing it once.
      2. ``close()`` clears the handle FIRST, so a running worker stops on its
         next check, and only then releases the underlying descriptor.
      3. ``close()`` waits for the worker to actually exit before releasing
         it, and an ``asyncio.Lock`` keeps a second exchange from starting
         during a close.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # Set whenever no worker is running. ``close()`` waits on it.
        self._idle = threading.Event()
        self._idle.set()
        self._closing = False

    async def open(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def exchange(self, frame: bytes, timeout: float) -> bytes:  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def _run(self, work: Callable[[], bytes]) -> bytes:
        """Run one exchange on a worker thread, one at a time.

        The idle flag is cleared here, before dispatch, and set INSIDE THE
        WORKER. That asymmetry is the point. Setting it in a ``finally`` on
        this coroutine would mark the device free the instant the await was
        cancelled — while the thread was still inside its read loop — which
        is precisely the case this whole mechanism exists to cover.
        """
        async with self._lock:
            if self._closing:
                return b""
            self._idle.clear()

            def _tracked() -> bytes:
                try:
                    return work()
                finally:
                    self._idle.set()

            return await asyncio.to_thread(_tracked)

    async def _quiesce(self) -> None:
        """Stop accepting work and wait for any worker to leave the device."""
        self._closing = True
        if not self._idle.is_set():
            # The await may have been cancelled while the thread ran on, so
            # this is the only thing that actually joins it.
            await asyncio.to_thread(self._idle.wait, _JOIN_TIMEOUT_SECONDS)
            if not self._idle.is_set():
                # Refusing to close is the safe answer: leaking one descriptor
                # is survivable, handing a live one to the next open is not.
                logger.error(
                    "voltronic_worker_did_not_stop",
                    msg=(
                        "the device I/O thread is still running after "
                        f"{_JOIN_TIMEOUT_SECONDS}s; leaving the handle open "
                        "rather than risking its reuse by another file"
                    ),
                )


class SerialTransport(_Transport):
    """RS232 or a USB-to-serial lead.

    pyserial is synchronous, so every call goes through a worker thread. A
    blocking read on the event loop would stall the collector's other
    adapters for the length of the inverter's timeout.
    """

    def __init__(self, port: str, baudrate: int) -> None:
        super().__init__()
        self._port = port
        self._baudrate = baudrate
        self._serial: Any = None

    async def open(self) -> None:
        # Imported here, not at module scope: the registry imports every
        # adapter at start-up, and a missing optional backend must not take
        # the whole collector down with it.
        import serial

        def _open() -> Any:
            return serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1.0,
                write_timeout=2.0,
            )

        self._serial = await asyncio.to_thread(_open)

    async def exchange(self, frame: bytes, timeout: float) -> bytes:
        if self._closing:
            # Closed, not broken. The caller loses this poll, not the node.
            return b""
        if self._serial is None:
            raise RuntimeError("serial transport is not open")

        def _exchange() -> bytes:
            port = self._serial
            if port is None:
                return b""
            port.reset_input_buffer()
            port.write(frame)
            port.flush()
            time.sleep(_SETTLE_SECONDS)
            deadline = time.monotonic() + timeout
            buffer = bytearray()
            while time.monotonic() < deadline:
                # Re-read every iteration: close() clears it, and this is
                # where the worker notices.
                port = self._serial
                if port is None:
                    break
                chunk = port.read(_READ_CHUNK)
                if chunk:
                    buffer.extend(chunk)
                    index = buffer.find(_TERMINATOR)
                    if index >= 0:
                        return bytes(buffer[: index + 1])
            return bytes(buffer)

        return await self._run(_exchange)

    async def close(self) -> None:
        # Clear the handle first so a running worker stops touching it, then
        # wait for that worker, and only then close the port.
        target, self._serial = self._serial, None
        await self._quiesce()
        if target is None or not self._idle.is_set():
            return
        await asyncio.to_thread(target.close)


class HidRawTransport(_Transport):
    """The inverter's own USB socket, which enumerates as a HID device.

    The protocol bytes are identical to the serial path. HID only changes
    how they are carried: writes go out in 8-byte reports, zero-padded, and
    the reply arrives in pieces that have to be reassembled up to the first
    carriage return.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path
        self._fd: int | None = None

    async def open(self) -> None:
        self._fd = await asyncio.to_thread(
            os.open, self._path, os.O_RDWR | os.O_NONBLOCK
        )

    async def exchange(self, frame: bytes, timeout: float) -> bytes:
        if self._closing:
            return b""
        if self._fd is None:
            raise RuntimeError("hidraw transport is not open")

        def _exchange() -> bytes:
            # Never capture the descriptor: a closed one is reissued to the
            # very next open, and writing to it then reaches another file.
            fd = self._fd
            if fd is None:
                return b""
            if len(frame) <= _HID_REPORT_BYTES:
                os.write(fd, frame)
            else:
                for start in range(0, len(frame), _HID_REPORT_BYTES):
                    chunk = frame[start : start + _HID_REPORT_BYTES]
                    if len(chunk) < _HID_REPORT_BYTES:
                        chunk = chunk + b"\x00" * (_HID_REPORT_BYTES - len(chunk))
                    time.sleep(_CHUNK_PAUSE_SECONDS)
                    fd = self._fd
                    if fd is None:
                        return b""
                    os.write(fd, chunk)
            time.sleep(_SETTLE_SECONDS)
            deadline = time.monotonic() + timeout
            buffer = bytearray()
            while time.monotonic() < deadline:
                fd = self._fd
                if fd is None:
                    break
                try:
                    chunk = os.read(fd, _READ_CHUNK)
                except BlockingIOError:
                    chunk = b""
                except OSError:
                    break
                if chunk:
                    buffer.extend(chunk)
                    index = buffer.find(_TERMINATOR)
                    if index >= 0:
                        return bytes(buffer[: index + 1])
                else:
                    time.sleep(0.15)
            return bytes(buffer)

        return await self._run(_exchange)

    async def close(self) -> None:
        fd, self._fd = self._fd, None
        await self._quiesce()
        if fd is None or not self._idle.is_set():
            return
        await asyncio.to_thread(os.close, fd)


def build_transport(config: dict[str, Any]) -> _Transport:
    """Pick a transport from the device path, or from an explicit override.

    A `/dev/hidraw*` path is the inverter's own USB socket; anything else is
    a serial line. The path is what decides, because that is what an
    installer actually knows.
    """
    port = str(config.get("port") or config.get("device") or "")
    if not port:
        raise ValueError("voltronic adapter needs a 'port' (e.g. /dev/ttyUSB0)")
    kind = str(config.get("transport") or "").lower()
    if not kind:
        kind = "hidraw" if "hidraw" in port else "serial"
    if kind == "hidraw":
        return HidRawTransport(port)
    if kind == "serial":
        return SerialTransport(port, int(config.get("baudrate", DEFAULT_BAUDRATE)))
    raise ValueError(f"unknown voltronic transport {kind!r} (serial or hidraw)")


class VoltronicInverterAdapter(HardwareAdapter):
    """Polls one Voltronic-core inverter over its own ASCII protocol."""

    def __init__(self) -> None:
        self._transport: _Transport | None = None
        self._status = AdapterStatus.DISCONNECTED
        self._generation = ""
        self._asset_ids: dict[str, str] = {}
        self._identity: dict[str, str] = {}
        self._timeout = 5.0
        self._verify_crc = True
        self._port_name = ""

    # ── lifecycle ───────────────────────────────────────────

    async def initialize(self, config: dict[str, Any]) -> bool:
        self._status = AdapterStatus.CONNECTING
        self._timeout = float(config.get("timeout_seconds", 5.0))
        # An escape hatch for a unit whose firmware computes the checksum
        # differently from both references. Off means the reply is trusted
        # without verification, so it is deliberately opt-in and logged.
        self._verify_crc = bool(config.get("verify_crc", True))
        if not self._verify_crc:
            logger.warning(
                "voltronic_crc_verification_disabled", port=config.get("port")
            )

        mappings = config.get("asset_mappings") or {}
        self._asset_ids = {str(k): str(v) for k, v in mappings.items()}
        if not self._asset_ids:
            logger.error(
                "voltronic_no_asset_mappings",
                msg="asset_mappings must bind inverter/battery/solar_array/grid_connection",
            )
            self._status = AdapterStatus.ERROR
            return False

        # A binding that is not a UUID is worse than no binding at all. The
        # adapter would start, log CONNECTED and collect happily; the rows
        # would reach SQLite, fail sync's pre-flight, be marked rejected and
        # pruned after the retention window. The series would be gone, and
        # the only trace a warning in a different container. Every other
        # role-binding adapter here refuses to start on this, so this one
        # does too.
        for role, asset_id in sorted(self._asset_ids.items()):
            try:
                uuid.UUID(asset_id)
            except (AttributeError, TypeError, ValueError):
                logger.error(
                    "voltronic_asset_mapping_not_a_uuid",
                    role=role,
                    asset_id=asset_id,
                    msg=(
                        "asset_mappings must bind each role to an asset UUID. "
                        "A non-UUID binding is silently discarded by sync and "
                        "the readings are lost, so this refuses to start"
                    ),
                )
                self._status = AdapterStatus.ERROR
                return False

        try:
            self._transport = build_transport(config)
            await self._transport.open()
        except Exception:
            logger.exception("voltronic_open_failed", port=config.get("port"))
            self._status = AdapterStatus.ERROR
            return False

        self._port_name = str(config.get("port") or "")
        self._generation = await self._detect_generation()
        if not self._generation:
            # The port opened but nothing answered. This is NOT a failure to
            # initialize, and that distinction is load-bearing: the collector
            # only registers a poll task for an adapter whose instance
            # exists, and its reconnect command only reaches an adapter that
            # has one. Returning False here would mean an inverter that was
            # merely asleep when the node booted — which is most of them, at
            # 03:00 on an off-grid site — never gets read again until someone
            # restarts the container. So the adapter starts, publishes
            # nothing, and retries detection on every poll until the unit
            # answers. Same shape as a Modbus adapter, whose connect() only
            # needs the port to open.
            logger.warning(
                "voltronic_protocol_not_detected",
                port=self._port_name,
                msg=(
                    "neither the PI30 nor the PI18 identity command was answered. "
                    "Publishing nothing and retrying on each poll. Check the lead "
                    "(2400 8N1 on serial) and that nothing else holds the port"
                ),
            )
            self._status = AdapterStatus.ERROR
            return True

        await self._read_identity()
        self._status = AdapterStatus.CONNECTED
        logger.info(
            "voltronic_connected",
            port=self._port_name,
            generation=self._generation,
            identity=self._identity,
        )
        return True

    async def _detect_generation(self) -> str:
        """Ask in PI30 framing first, then PI18.

        There is no command that works across the family — neither reference
        implementation autodetects at all, and mpp-solar makes the protocol a
        required argument. So detection is: try a framing, see if a valid
        reply comes back.
        """
        reply = await self._send_pi30("QPI")
        if reply is not None and reply.fields:
            # A PI30 unit answers "(PI30"; a PI17 unit answers "(PI17" in the
            # same envelope and reads the same way.
            return "pi30"

        reply = await self._send_pi18("PI")
        if reply is not None and reply.fields:
            return "pi18"
        return ""

    async def _read_identity(self) -> None:
        """Serial and firmware, kept as evidence rather than telemetry."""
        if self._generation == "pi30":
            serial_reply = await self._send_pi30("QID")
            firmware_reply = await self._send_pi30("QVFW")
        else:
            serial_reply = await self._send_pi18("ID")
            firmware_reply = await self._send_pi18("VFW")
        if serial_reply is not None and serial_reply.fields:
            self._identity["serial"] = serial_reply.fields[0].strip()
        if firmware_reply is not None and firmware_reply.fields:
            self._identity["firmware"] = firmware_reply.fields[0].strip()
        self._identity["protocol"] = self._generation

    def identity(self) -> dict[str, str]:
        return dict(self._identity)

    def vendor_map_summary(self) -> dict[str, Any] | None:
        """Provenance and identity, shipped once as tier-3 evidence.

        Named for the vendor-map contract the collector looks for, because
        that is the hook — but this adapter has no map. What a validation
        record needs is the same either way: which code read the device, on
        what evidence, and what the unit said it was. For a Voltronic unit
        that is the protocol generation it turned out to speak, plus its
        serial and firmware.

        Without this the adapter would be the only local one whose unit
        identity never left the node, which would make a Voltronic validation
        record strictly weaker than every other one.

        Offline-first: this is a pure accessor. The collector enqueues it.
        """
        if not self._generation:
            return None  # nothing identified yet; poll() will try again
        return {
            "name": f"voltronic_{self._generation}",
            "vendor": "voltronic",
            "model_family": "Axpert-family core (PI30/PI18 command protocol)",
            # Deliberately the same tier vocabulary as the maps. The adapter
            # implements a published protocol; no physical unit has been read.
            "verification": "community",
            "source": (
                "jblance/mpp-solar @eafdd432 (MIT); vendor document "
                '"HS_MS_MSX RS232 Protocol" 2014-08-22; the C reference in '
                "ned-kelly/docker-voltronic-homeassistant @ac264263; "
                "read 2026-09-19"
            ),
            "extends": [],
            # Not Modbus: there is no address space and no register map, and
            # saying so explicitly beats omitting the keys a reader expects.
            "protocol": self._generation,
            "transport": self._port_name,
            "register_map": None,
            "identity": dict(self._identity),
        }

    # ── exchanges ───────────────────────────────────────────

    async def _send_pi30(self, command: str) -> Reply | None:
        return await self._send(frame_pi30(command), parse_pi30_reply, command)

    async def _send_pi18(self, command: str) -> Reply | None:
        return await self._send(frame_pi18(command), parse_pi18_reply, command)

    async def _send(
        self,
        frame: bytes,
        parser: Callable[..., Reply],
        command: str,
    ) -> Reply | None:
        transport = self._transport
        if transport is None:
            return None
        try:
            raw = await transport.exchange(frame, self._timeout)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("voltronic_exchange_failed", command=command, exc_info=True)
            return None
        if not raw:
            logger.debug("voltronic_no_reply", command=command)
            return None
        try:
            return parser(raw, verify_crc=self._verify_crc)
        except VoltronicProtocolError as exc:
            logger.warning("voltronic_bad_reply", command=command, reason=str(exc))
            return None

    # ── polling ─────────────────────────────────────────────

    async def poll(self) -> list[Reading]:
        if self._transport is None:
            return []
        if not self._generation:
            # An inverter that was asleep at start-up gets picked up here,
            # without needing a container restart.
            self._generation = await self._detect_generation()
            if not self._generation:
                self._status = AdapterStatus.ERROR
                return []
            await self._read_identity()
            logger.info(
                "voltronic_connected",
                port=self._port_name,
                generation=self._generation,
                identity=self._identity,
                recovered=True,
            )
        try:
            values = (
                await self._poll_pi30()
                if self._generation == "pi30"
                else await self._poll_pi18()
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("voltronic_poll_failed", generation=self._generation)
            self._status = AdapterStatus.ERROR
            return []

        if not values:
            self._status = AdapterStatus.ERROR
            return []
        self._status = AdapterStatus.CONNECTED
        return self._to_readings(values)

    async def _poll_pi30(self) -> dict[str, float]:
        values: dict[str, float] = {}
        status = await self._send_pi30("QPIGS")
        if status is not None:
            values.update(parse_qpigs(status.fields))
        mode = await self._send_pi30("QMOD")
        if mode is not None:
            values.update(parse_qmod(mode.fields))
        warnings = await self._send_pi30("QPIWS")
        if warnings is not None:
            values.update(parse_qpiws(warnings.fields))
        return values

    async def _poll_pi18(self) -> dict[str, float]:
        values: dict[str, float] = {}
        status = await self._send_pi18("GS")
        if status is not None:
            values.update(parse_pi18_gs(status.fields))
        mode = await self._send_pi18("MOD")
        if mode is not None:
            values.update(parse_pi18_mod(mode.fields))
        # PI18's fault-and-warning command is not implemented here. V1 carries
        # it only as a commented-out stub, so there is no field order to
        # transcribe and nothing to check a guess against.
        return values

    #: Which bound asset each metric namespace belongs to. One inverter
    #: reports as four assets, the same shape the hybrid vendor maps use.
    _ROLE_BY_PREFIX = {
        "grid.": "grid_connection",
        "battery.": "battery",
        "solar.": "solar_array",
        "inverter.": "inverter",
    }

    def _to_readings(self, values: dict[str, float]) -> list[Reading]:
        now = datetime.now(UTC)
        readings: list[Reading] = []
        for metric, value in sorted(values.items()):
            role = next(
                (r for p, r in self._ROLE_BY_PREFIX.items() if metric.startswith(p)),
                "",
            )
            asset_id = self._asset_ids.get(role, "")
            if not asset_id:
                # An unbound role is dropped rather than published against
                # the wrong asset or against an empty one.
                continue
            readings.append(
                Reading(
                    metric_name=metric,
                    value=value,
                    timestamp=now,
                    quality=0,
                    raw_register=f"voltronic:{self._generation}",
                    asset_id=asset_id,
                )
            )
        return readings

    # ── the rest of the ABC ─────────────────────────────────

    async def subscribe(
        self, callback: Callable[[list[Reading]], Awaitable[None]]
    ) -> bool:
        # Polled request/response only; the inverter never pushes.
        return False

    async def write(self, metric_name: str, value: float) -> bool:
        logger.warning(
            "voltronic_write_refused",
            metric=metric_name,
            msg=(
                "this adapter is read-only. Voltronic setter commands are "
                "indistinguishable from queries on the wire, so only an explicit "
                "read allowlist is ever framed"
            ),
        )
        return False

    async def health_check(self) -> AdapterStatus:
        if self._transport is None:
            return AdapterStatus.DISCONNECTED
        probe = "QPI" if self._generation == "pi30" else "PI"
        reply = (
            await self._send_pi30(probe)
            if self._generation == "pi30"
            else await self._send_pi18(probe)
        )
        self._status = (
            AdapterStatus.CONNECTED if reply is not None else AdapterStatus.ERROR
        )
        return self._status

    async def shutdown(self) -> None:
        transport, self._transport = self._transport, None
        if transport is not None:
            try:
                await transport.close()
            except Exception:
                logger.debug("voltronic_close_failed", exc_info=True)
        self._status = AdapterStatus.DISCONNECTED

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            can_read=True,
            can_write=False,
            can_subscribe=False,
            # The unit is half-duplex at 2400 baud and a poll is three
            # round trips, each with a quarter-second settle. Below about
            # 20 seconds the exchanges start overlapping.
            poll_interval_seconds=30,
            min_poll_interval_seconds=20,
            supported_metrics=list(SUPPORTED_METRICS),
            supported_write_metrics=[],
            supports_asset_targeting=True,
        )
