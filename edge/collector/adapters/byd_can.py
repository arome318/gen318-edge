"""BYD Battery-Box CAN bus adapter (Doc 15 §7.4, F-HW-07).

Reads BYD BMS data via CAN bus (USB-CAN adapter, e.g., Canable, PEAK PCAN-USB).
Protocol: CANopen / proprietary BYD BMS protocol at 500 kbps.

Uses python-can's AsyncBufferedReader + Notifier for non-blocking
async CAN reads instead of blocking bus.recv().
"""

from __future__ import annotations

import asyncio
import struct
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import structlog

from .base import AdapterCapabilities, AdapterStatus, HardwareAdapter, Reading
from .victron_can_profile import VICTRON_FRAMES

logger = structlog.get_logger()


# ── CAN Message ID Definitions ───────────────────────────────
# BYD Battery-Box Premium HVS/HVM CAN protocol

# The 0x351/0x355/0x356/0x35E/0x35F set is the Victron-compatible BMS
# profile, shared with every other pack that speaks it, so it now lives in
# victron_can_profile. The layouts are byte-identical to what this module has
# been shipping — the BYD contract does not move. BYD's own contribution is
# the per-module cell frames below, which no other vendor sends.
CAN_MESSAGES: dict[int, dict[str, Any]] = VICTRON_FRAMES

# Cell voltage messages: 0x370-0x37F (per-module, up to 16 modules)
CELL_VOLTAGE_BASE = 0x370
CELL_VOLTAGE_COUNT = 16

# Cell temperature messages: 0x380-0x38F (per-module, up to 16 modules)
CELL_TEMP_BASE = 0x380
CELL_TEMP_COUNT = 16


class BYDCANAdapter(HardwareAdapter):
    """Reads BYD Battery-Box BMS data via CAN bus.

    Derives cell min/max voltage and temperature from per-module messages.
    Uses can.Notifier + can.AsyncBufferedReader for async non-blocking reads.
    """

    def __init__(self) -> None:
        self._bus: Any = None  # can.Bus (optional import)
        self._notifier: Any = None  # can.Notifier
        self._reader: Any = None  # can.AsyncBufferedReader
        self._config: dict[str, Any] = {}
        self._status = AdapterStatus.DISCONNECTED
        self._last_reading_at: float = 0.0
        self._asset_id: str = ""
        self._cell_voltages: list[float] = []
        self._cell_temps: list[float] = []

    async def initialize(self, config: dict[str, Any]) -> bool:
        """Open CAN bus interface and set up async reader."""
        self._config = config
        self._asset_id = config["asset_id"]

        try:
            import can

            self._bus = can.Bus(
                interface="socketcan",
                channel=config.get("interface", "can0"),
                bitrate=config.get("bitrate", 500000),
            )

            self._reader = can.AsyncBufferedReader()
            loop = asyncio.get_running_loop()
            self._notifier = can.Notifier(self._bus, [self._reader], loop=loop)

            self._status = AdapterStatus.CONNECTED
            logger.info(
                "byd_can_initialized",
                interface=config.get("interface", "can0"),
                bitrate=config.get("bitrate", 500000),
                asset_id=self._asset_id,
            )
            return True

        except Exception:
            logger.exception("byd_can_init_failed")
            self._status = AdapterStatus.ERROR
            return False

    async def poll(self) -> list[Reading]:
        """Read available CAN messages and return normalized readings.

        Collects messages for up to 2 seconds, then derives cell
        min/max metrics from accumulated per-module data.
        """
        if not self._reader:
            return []

        readings: list[Reading] = []
        now = datetime.now(UTC)
        deadline = time.time() + 2.0

        # Reset accumulated cell data each poll cycle to prevent stale
        # readings after BMS reconnect or module removal
        self._cell_voltages = []
        self._cell_temps = []

        while time.time() < deadline:
            try:
                msg = await asyncio.wait_for(self._reader.get_message(), timeout=0.5)
                if msg is not None:
                    try:
                        parsed = self._parse_message(msg, now)
                        readings.extend(parsed)
                    except (struct.error, IndexError) as exc:
                        logger.warning(
                            "byd_can_parse_error",
                            arb_id=f"0x{msg.arbitration_id:03X}"
                            if hasattr(msg, "arbitration_id")
                            else "unknown",
                            error=str(exc),
                        )
            except asyncio.TimeoutError:
                break

        # Derive min/max cell metrics from accumulated data
        readings.extend(self._derive_cell_metrics(now))

        if readings:
            self._last_reading_at = time.time()
            logger.debug("byd_can_poll_complete", reading_count=len(readings))

        return readings

    def _parse_message(self, msg: Any, now: datetime) -> list[Reading]:
        """Parse a single CAN message into readings."""
        readings: list[Reading] = []
        arb_id: int = msg.arbitration_id
        data: bytes = msg.data

        # Standard BMS messages
        if arb_id in CAN_MESSAGES:
            msg_def = CAN_MESSAGES[arb_id]
            for metric_name, field_def in msg_def["fields"].items():
                offset: int = field_def["offset"]
                length: int = field_def["length"]
                scale: float = field_def["scale"]
                signed: bool = field_def["signed"]

                raw_bytes = data[offset : offset + length]
                if length == 2:
                    fmt = "<h" if signed else "<H"
                    raw_value = struct.unpack(fmt, raw_bytes)[0]
                elif length == 1:
                    fmt = "<b" if signed else "<B"
                    raw_value = struct.unpack(fmt, raw_bytes)[0]
                else:
                    continue

                value = raw_value * scale

                readings.append(
                    Reading(
                        metric_name=metric_name,
                        value=value,
                        timestamp=now,
                        quality=0,
                        raw_register=f"can:0x{arb_id:03X}:{offset}",
                        asset_id=self._asset_id,
                    )
                )

        # Cell voltage messages (0x370-0x37F)
        elif CELL_VOLTAGE_BASE <= arb_id < CELL_VOLTAGE_BASE + CELL_VOLTAGE_COUNT:
            module_idx = arb_id - CELL_VOLTAGE_BASE
            for i in range(0, min(len(data), 8), 2):
                cell_v = struct.unpack("<H", data[i : i + 2])[0] * 0.001  # mV to V
                if cell_v > 0:
                    cell_idx = module_idx * 4 + (i // 2)
                    if cell_idx >= len(self._cell_voltages):
                        self._cell_voltages.extend(
                            [0.0] * (cell_idx - len(self._cell_voltages) + 1)
                        )
                    self._cell_voltages[cell_idx] = cell_v

        # Cell temperature messages (0x380-0x38F)
        elif CELL_TEMP_BASE <= arb_id < CELL_TEMP_BASE + CELL_TEMP_COUNT:
            module_idx = arb_id - CELL_TEMP_BASE
            for i in range(0, min(len(data), 8), 2):
                cell_t = struct.unpack("<h", data[i : i + 2])[0] * 0.1
                cell_idx = module_idx * 4 + (i // 2)
                if cell_idx >= len(self._cell_temps):
                    self._cell_temps.extend(
                        [0.0] * (cell_idx - len(self._cell_temps) + 1)
                    )
                self._cell_temps[cell_idx] = cell_t

        return readings

    def _derive_cell_metrics(self, now: datetime) -> list[Reading]:
        """Derive min/max cell voltage and temperature from accumulated data."""
        derived: list[Reading] = []

        active_voltages = [v for v in self._cell_voltages if v > 0]
        if active_voltages:
            derived.append(
                Reading(
                    metric_name="battery.cell_min_voltage_v",
                    value=min(active_voltages),
                    timestamp=now,
                    quality=0,
                    raw_register="derived:cell_voltages",
                    asset_id=self._asset_id,
                )
            )
            derived.append(
                Reading(
                    metric_name="battery.cell_max_voltage_v",
                    value=max(active_voltages),
                    timestamp=now,
                    quality=0,
                    raw_register="derived:cell_voltages",
                    asset_id=self._asset_id,
                )
            )

        if self._cell_temps:
            derived.append(
                Reading(
                    metric_name="battery.cell_min_temp_c",
                    value=min(self._cell_temps),
                    timestamp=now,
                    quality=0,
                    raw_register="derived:cell_temps",
                    asset_id=self._asset_id,
                )
            )
            derived.append(
                Reading(
                    metric_name="battery.cell_max_temp_c",
                    value=max(self._cell_temps),
                    timestamp=now,
                    quality=0,
                    raw_register="derived:cell_temps",
                    asset_id=self._asset_id,
                )
            )

        return derived

    async def subscribe(
        self,
        callback: Callable[[list[Reading]], Awaitable[None]],
    ) -> bool:
        """CAN bus is inherently push-based; wrap in continuous read loop."""
        while True:
            readings = await self.poll()
            if readings:
                await callback(readings)
            await asyncio.sleep(1)

    async def write(self, metric_name: str, value: float) -> bool:
        """BYD CAN bus is read-only for monitoring purposes."""
        return False

    async def health_check(self) -> AdapterStatus:
        """Return ERROR if no readings in 2 minutes."""
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
            supported_metrics=[
                "battery.soc_pct",
                "battery.soh_pct",
                "battery.voltage_v",
                "battery.current_a",
                "battery.temperature_c",
                "battery.cell_min_voltage_v",
                "battery.cell_max_voltage_v",
                "battery.cell_min_temp_c",
                "battery.cell_max_temp_c",
                "battery.charge_voltage_v",
                "battery.charge_current_limit_a",
                "battery.discharge_current_limit_a",
            ],
            poll_interval_seconds=1,
            min_poll_interval_seconds=1,
        )

    async def shutdown(self) -> None:
        """Stop notifier and close CAN bus."""
        if self._notifier:
            self._notifier.stop()
        if self._bus:
            self._bus.shutdown()
        self._status = AdapterStatus.DISCONNECTED
        logger.info("byd_can_shutdown", asset_id=self._asset_id)
