"""Victron Modbus TCP adapter for direct register-level access.

Reads all registers for Unit IDs 100 (System), 225 (Battery Monitor),
246 (VE.Bus/Multi/Quattro), and 247 (Solar Charger/MPPT) via the
GX device's Modbus TCP server on port 502.

Uses pymodbus built-in reconnect_delay/reconnect_delay_max instead
of rolling custom reconnection logic.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import structlog
from pymodbus.client import AsyncModbusTcpClient
from pymodbus.exceptions import ModbusIOException
from pymodbus.framer import FramerType

from .base import AdapterCapabilities, AdapterStatus, HardwareAdapter, Reading
from .generic_modbus import DEVICE_KWARG

logger = structlog.get_logger()


# ── Complete Modbus Register Map ─────────────────────────────

MODBUS_REGISTERS: dict[int, dict[str, dict[str, Any]]] = {
    # Unit ID 100 — System (com.victronenergy.system)
    100: {
        "battery.voltage_v": {"reg": 840, "type": "uint16", "scale": 10},
        "battery.current_a": {"reg": 841, "type": "int16", "scale": 10},
        "battery.power_w": {"reg": 842, "type": "int16", "scale": 1},
        "battery.soc_pct": {"reg": 843, "type": "uint16", "scale": 1},
        "battery.state": {"reg": 844, "type": "uint16", "scale": 1},
        "battery.consumed_ah": {"reg": 845, "type": "uint16", "scale": -10},
        "battery.time_to_go_s": {"reg": 846, "type": "uint16", "scale": 0.01},
        "solar.pv_power_w": {"reg": 850, "type": "uint16", "scale": 1},
        "solar.pv_current_a": {"reg": 851, "type": "int16", "scale": 10},
        "system.dc_system_power_w": {"reg": 860, "type": "int16", "scale": 1},
        "system.ac_consumption_w": {"reg": 817, "type": "uint16", "scale": 1},
        "system.grid_power_w": {"reg": 820, "type": "int16", "scale": 1},
        "genset.power_w": {"reg": 823, "type": "int16", "scale": 1},
    },
    # Unit ID 246 — VE.Bus / Multi / Quattro
    246: {
        "inverter.ac_input_voltage_v": {"reg": 3, "type": "uint16", "scale": 10},
        "inverter.ac_input_current_a": {"reg": 6, "type": "int16", "scale": 10},
        "inverter.ac_input_freq_hz": {"reg": 9, "type": "int16", "scale": 100},
        "inverter.ac_input_power_w": {"reg": 12, "type": "int16", "scale": 0.1},
        "inverter.ac_output_voltage_v": {"reg": 15, "type": "uint16", "scale": 10},
        "inverter.ac_output_current_a": {"reg": 18, "type": "int16", "scale": 10},
        "inverter.ac_output_freq_hz": {"reg": 21, "type": "int16", "scale": 100},
        "inverter.ac_output_power_w": {"reg": 23, "type": "int16", "scale": 0.1},
        "battery.voltage_v": {"reg": 26, "type": "uint16", "scale": 100},
        "battery.current_a": {"reg": 27, "type": "int16", "scale": 10},
        "battery.soc_pct": {"reg": 30, "type": "uint16", "scale": 10},
        "inverter.state": {"reg": 31, "type": "uint16", "scale": 1},
        "inverter.error_code": {"reg": 32, "type": "uint16", "scale": 1},
        "inverter.temperature_c": {"reg": 61, "type": "int16", "scale": 10},
    },
    # Unit ID 247 — Solar Charger (MPPT)
    247: {
        "battery.voltage_v": {"reg": 771, "type": "uint16", "scale": 100},
        "battery.current_a": {"reg": 772, "type": "int16", "scale": 10},
        "solar.charger_state": {"reg": 775, "type": "uint16", "scale": 1},
        "solar.pv_voltage_v": {"reg": 776, "type": "uint16", "scale": 100},
        "solar.pv_current_a": {"reg": 777, "type": "int16", "scale": 10},
        "solar.daily_yield_kwh": {"reg": 784, "type": "uint16", "scale": 10},
        "solar.max_power_today_w": {"reg": 785, "type": "uint16", "scale": 1},
        "solar.pv_power_w": {"reg": 789, "type": "uint16", "scale": 10},
        "solar.total_yield_kwh": {"reg": 790, "type": "uint16", "scale": 10},
    },
    # Unit ID 225 — Battery Monitor / CAN-bus BMS
    225: {
        "battery.voltage_v": {"reg": 259, "type": "uint16", "scale": 100},
        "battery.current_a": {"reg": 261, "type": "int16", "scale": 10},
        "battery.power_w": {"reg": 258, "type": "int16", "scale": 1},
        "battery.temperature_c": {"reg": 262, "type": "int16", "scale": 10},
        "battery.cell_min_voltage_v": {"reg": 263, "type": "uint16", "scale": 100},
        "battery.consumed_ah": {"reg": 265, "type": "uint16", "scale": -10},
        "battery.soc_pct": {"reg": 266, "type": "uint16", "scale": 10},
        # /History/ChargeCycles (BMV H4): discrete cycle counter — counts a
        # discharge below ~65% followed by a recharge above ~90%. NOT
        # equivalent-full-cycles. Documented in Victron dbus_modbustcp
        # attributes.csv; canonical cloud metric battery.cycles (COUNTER).
        "battery.cycles": {"reg": 284, "type": "uint16", "scale": 1},
        "battery.time_to_go_s": {"reg": 303, "type": "uint16", "scale": 0.01},
        "battery.soh_pct": {"reg": 304, "type": "uint16", "scale": 10},
        "battery.capacity_ah": {"reg": 309, "type": "uint16", "scale": 10},
    },
}

# Bounds for basic reading validation
_BOUNDS: dict[str, tuple[float, float]] = {
    "battery.cycles": (0, 10_000_000),
    "battery.soc_pct": (0, 100),
    "battery.voltage_v": (0, 100),
    "battery.temperature_c": (-20, 80),
    "solar.pv_power_w": (0, 200_000),
    "inverter.efficiency_pct": (0, 100),
    "inverter.temperature_c": (-20, 100),
}


class VictronModbusAdapter(HardwareAdapter):
    """Reads Victron equipment via Modbus TCP on the GX device (port 502).

    Supports all four Unit IDs with complete register maps. Uses
    pymodbus built-in auto-reconnect with exponential backoff.
    """

    def __init__(self) -> None:
        self._client: AsyncModbusTcpClient | None = None
        self._config: dict[str, Any] = {}
        self._status = AdapterStatus.DISCONNECTED
        self._last_reading_at: float = 0.0
        self._error_count: int = 0

    async def initialize(self, config: dict[str, Any]) -> bool:
        """Connect to the Victron GX device via Modbus TCP."""
        self._config = config
        timeout_s = config.get("timeout_ms", 3000) / 1000.0

        self._client = AsyncModbusTcpClient(
            host=config["host"],
            port=config.get("port", 502),
            timeout=timeout_s,
            retries=config.get("retries", 3),
            reconnect_delay=1,
            reconnect_delay_max=60,
            framer=FramerType.SOCKET,
        )

        connected = await self._client.connect()
        if connected:
            self._status = AdapterStatus.CONNECTED
            self._error_count = 0
            logger.info(
                "victron_modbus_initialized",
                host=config["host"],
                port=config.get("port", 502),
                unit_ids=config.get("unit_ids", [100, 225, 246, 247]),
            )
        else:
            self._status = AdapterStatus.ERROR

        return connected

    async def poll(self) -> list[Reading]:
        """Read all configured registers across all unit IDs."""
        if not self._client or not self._client.connected:
            return []

        readings: list[Reading] = []
        now = datetime.now(UTC)
        unit_ids = self._config.get("unit_ids", [100, 225, 246, 247])
        asset_mappings = self._config.get("asset_mappings", {})

        for unit_id in unit_ids:
            if unit_id not in MODBUS_REGISTERS:
                continue

            asset_id = asset_mappings.get(str(unit_id), f"unit-{unit_id}")
            registers = MODBUS_REGISTERS[unit_id]

            for metric_name, reg_def in registers.items():
                try:
                    value = await self._read_register(
                        unit_id=unit_id,
                        address=reg_def["reg"],
                        data_type=reg_def["type"],
                        scale=reg_def["scale"],
                    )
                    if value is not None:
                        quality = self._validate_reading(metric_name, value)
                        readings.append(
                            Reading(
                                metric_name=metric_name,
                                value=value,
                                timestamp=now,
                                quality=quality,
                                raw_register=f"uid:{unit_id}/reg:{reg_def['reg']}",
                                asset_id=asset_id,
                            )
                        )
                except ModbusIOException:
                    self._error_count += 1

        if readings:
            self._last_reading_at = time.time()
            self._error_count = 0
            logger.debug(
                "victron_modbus_poll_complete",
                reading_count=len(readings),
                unit_ids=unit_ids,
            )

        return readings

    async def _read_register(
        self,
        unit_id: int,
        address: int,
        data_type: str,
        scale: float,
    ) -> float | None:
        """Read a single Modbus register and apply scaling."""
        assert self._client is not None

        count = 2 if data_type == "uint32" else 1
        result = await self._client.read_holding_registers(
            address=address,
            count=count,
            **{DEVICE_KWARG: unit_id},
        )

        if result.isError():
            return None

        raw_value = result.registers[0]

        if data_type == "int16":
            if raw_value > 32767:
                raw_value -= 65536
        elif data_type == "uint32":
            raw_value = (result.registers[0] << 16) | result.registers[1]

        # Apply scale:
        # scale >= 1 → divide (register stores value * scale)
        # 0 < scale < 1 → multiply (register stores value / scale)
        # scale < 0 → divide by absolute value
        # scale == 0 → return 0
        if scale == 0:
            return 0.0
        elif scale < 0:
            return raw_value / abs(scale)
        elif scale >= 1:
            return raw_value / scale
        else:
            return raw_value * scale

    @staticmethod
    def _validate_reading(metric_name: str, value: float) -> int:
        """Basic bounds checking. Returns quality: 0=good, 1=suspect."""
        if metric_name in _BOUNDS:
            lo, hi = _BOUNDS[metric_name]
            if value < lo or value > hi:
                return 1  # suspect
        return 0  # good

    async def subscribe(
        self,
        callback: Callable[[list[Reading]], Awaitable[None]],
    ) -> bool:
        """Modbus TCP is poll-based; wrap poll in a timed loop.

        Raises RuntimeError if the client disconnects so the supervisor
        can restart the adapter instead of silently producing no data.
        """
        interval = self._config.get("poll_interval_seconds", 60)
        consecutive_empty = 0
        while True:
            readings = await self.poll()
            if readings:
                await callback(readings)
                consecutive_empty = 0
            else:
                consecutive_empty += 1
                # If client disconnected and poll keeps returning empty,
                # raise to trigger supervisor restart
                if (
                    self._client
                    and not self._client.connected
                    and consecutive_empty >= 3
                ):
                    raise RuntimeError(
                        "Modbus client disconnected — triggering supervisor restart"
                    )
            await asyncio.sleep(interval)

    async def write(self, metric_name: str, value: float) -> bool:
        """Write only an explicitly commissioned control register."""
        if not self._client or not self._client.connected:
            return False
        mappings = self._config.get("write_mappings") or {}
        reg_def = mappings.get(metric_name) if isinstance(mappings, dict) else None
        if not isinstance(reg_def, dict):
            return False
        scale = float(reg_def.get("scale", 1.0))
        if scale == 0:
            return False
        raw_value = int(round(value / scale))
        if str(reg_def.get("type", "uint16")) == "int16" and raw_value < 0:
            raw_value += 65536
        if not 0 <= raw_value <= 65535:
            return False
        result = await self._client.write_register(
            address=int(reg_def["address"]),
            value=raw_value,
            **{DEVICE_KWARG: int(reg_def.get("unit_id", 100))},
        )
        return not result.isError()

    async def health_check(self) -> AdapterStatus:
        """Return status based on connection state and error count."""
        if not self._client or not self._client.connected:
            return AdapterStatus.DISCONNECTED
        if self._error_count > 10:
            return AdapterStatus.ERROR
        if self._last_reading_at > 0 and time.time() - self._last_reading_at > 300:
            return AdapterStatus.ERROR
        return AdapterStatus.CONNECTED

    def capabilities(self) -> AdapterCapabilities:
        all_metrics: list[str] = []
        for unit_id in self._config.get("unit_ids", []):
            if unit_id in MODBUS_REGISTERS:
                all_metrics.extend(MODBUS_REGISTERS[unit_id].keys())
        return AdapterCapabilities(
            can_read=True,
            can_write=bool(self._config.get("write_mappings")),
            can_subscribe=False,
            supported_metrics=list(set(all_metrics)),
            supported_write_metrics=sorted(
                (self._config.get("write_mappings") or {}).keys()
            ),
            poll_interval_seconds=self._config.get("poll_interval_seconds", 60),
            min_poll_interval_seconds=5,
        )

    async def shutdown(self) -> None:
        """Close the Modbus TCP client."""
        if self._client:
            self._client.close()
        self._status = AdapterStatus.DISCONNECTED
        logger.info("victron_modbus_shutdown")
