"""Edge hardware adapters package (Doc 15).

Provides concrete HardwareAdapter implementations for edge-side
hardware communication: Victron MQTT/Modbus, BYD CAN bus, Okra Solar,
and a generic YAML-configured Modbus adapter.
"""

from .base import AdapterCapabilities, AdapterStatus, HardwareAdapter, Reading
from .registry import ADAPTER_REGISTRY, create_adapter

__all__ = [
    "AdapterCapabilities",
    "AdapterStatus",
    "HardwareAdapter",
    "Reading",
    "ADAPTER_REGISTRY",
    "create_adapter",
]
