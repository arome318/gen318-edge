"""Adapter registry — maps config type strings to adapter classes.

Pattern mirrors backend/app/adapters/registry.py for consistency.
"""

from __future__ import annotations

from typing import Any

import structlog

from .base import HardwareAdapter
from .byd_can import BYDCANAdapter
from .can_bms_listener import CANBMSListenerAdapter
from .generic_modbus import GenericModbusAdapter
from .okra_harvest import OkraHarvestAdapter
from .sunspec_modbus import SunSpecModbusAdapter
from .victron_modbus import VictronModbusAdapter
from .victron_mqtt import VictronMQTTAdapter
from .voltronic_inverter import VoltronicInverterAdapter

logger = structlog.get_logger()

ADAPTER_REGISTRY: dict[str, type[HardwareAdapter]] = {
    "victron_mqtt": VictronMQTTAdapter,
    "victron_modbus": VictronModbusAdapter,
    "byd_can": BYDCANAdapter,
    # Receive-only tap on an inverter-to-battery bus. Never transmits, and
    # refuses to start unless the interface is already listen-only.
    "can_bms_listener": CANBMSListenerAdapter,
    "okra_harvest": OkraHarvestAdapter,
    "generic_modbus": GenericModbusAdapter,
    "sunspec_modbus": SunSpecModbusAdapter,
    # Not Modbus: Voltronic's own ASCII command protocol, over serial or
    # USB HID. Read-only by construction — its setter commands look
    # identical to its queries, so only an allowlist is ever framed.
    "voltronic_inverter": VoltronicInverterAdapter,
}


async def create_adapter(adapter_type: str, config: dict[str, Any]) -> HardwareAdapter:
    """Create and initialize an adapter instance from the registry.

    Raises ``ValueError`` if the adapter type is unknown.
    Raises ``RuntimeError`` if initialization fails.
    """
    cls = ADAPTER_REGISTRY.get(adapter_type)
    if cls is None:
        raise ValueError(
            f"Unknown adapter type: {adapter_type!r}. "
            f"Available: {list(ADAPTER_REGISTRY.keys())}"
        )

    adapter = cls()
    ok = await adapter.initialize(config)
    if not ok:
        raise RuntimeError(f"Adapter {adapter_type!r} failed to initialize")

    logger.info("adapter_created", adapter_type=adapter_type)
    return adapter
