"""Hardware adapter abstract interface for edge node (Doc 15).

Mirrors backend/app/adapters/base.py for edge self-containment.
The edge runs in Docker containers on RPi 5 with no access to the
backend Python package. Keep signatures in sync with the backend copy.

Edge extension: Reading includes an ``asset_id`` field for direct
asset mapping at the adapter level, avoiding fragile raw_register parsing.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


class AdapterStatus(enum.Enum):
    """Lifecycle state of an adapter instance."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class Reading:
    """A single metric observation from a hardware adapter.

    The ``asset_id`` field maps this reading to a Gen318 asset UUID.
    Each adapter resolves asset IDs from its config's ``asset_mappings``.
    """

    metric_name: str
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    quality: int = 0  # 0=good, 1=suspect, 2=bad
    raw_register: str = ""  # original register/topic for debugging
    asset_id: str = ""  # Gen318 asset UUID


@dataclass
class AdapterCapabilities:
    """Declares what an adapter can and cannot do."""

    can_read: bool = True
    can_write: bool = False
    can_subscribe: bool = False
    poll_interval_seconds: int = 60
    min_poll_interval_seconds: int = 5
    supported_metrics: list[str] = field(default_factory=list)
    supported_write_metrics: list[str] = field(default_factory=list)
    supports_asset_targeting: bool = False
    # Historical backfill — mirrors the backend ABC. Edge local-bus adapters
    # (Modbus/CAN/MQTT) have no vendor history API, so these stay defaults.
    can_backfill: bool = False
    max_history_days: int = 0
    history_chunk_days: int = 0


class HistoryFetchError(Exception):
    """A historical fetch failed for transient/permission reasons.

    Mirrors the backend ABC contract: fetch_history raises this instead of
    returning [] on failure, so [] always means "no data in window".
    """


class HardwareAdapter(ABC):
    """Base class for all edge hardware adapters.

    Each adapter normalizes vendor-specific telemetry to canonical metric
    names and populates ``Reading.asset_id`` from its configuration.
    """

    @abstractmethod
    async def initialize(self, config: dict[str, Any]) -> bool:
        """Connect to the hardware or cloud API.

        Returns ``True`` on success, ``False`` on failure.
        """
        ...

    @abstractmethod
    async def poll(self) -> list[Reading]:
        """Fetch the latest telemetry readings.

        Returns a list of Reading instances with canonical metric names.
        Returns an empty list on error (graceful degradation).
        """
        ...

    @abstractmethod
    async def subscribe(
        self,
        callback: Callable[[list[Reading]], Awaitable[None]],
    ) -> bool:
        """Register a push-based callback for real-time data.

        Returns ``True`` if subscriptions were established successfully.
        """
        ...

    @abstractmethod
    async def write(self, metric_name: str, value: float) -> bool:
        """Write a set-point or command to the hardware."""
        ...

    async def fetch_history(
        self,
        start: datetime,
        end: datetime,
        metrics: list[str] | None = None,
    ) -> list[Reading]:
        """Fetch historical readings for the window [start, end).

        Concrete default mirroring the backend ABC: edge adapters have no
        cloud history behind Modbus/CAN, so the default empty list applies.
        Implementations must raise HistoryFetchError on fetch failure —
        [] always means "no data in window".
        """
        return []

    @abstractmethod
    async def health_check(self) -> AdapterStatus:
        """Probe connectivity and return the current adapter status."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources and close connections."""
        ...

    @abstractmethod
    def capabilities(self) -> AdapterCapabilities:
        """Return the static capabilities descriptor for this adapter."""
        ...
