"""Okra Solar mesh-grid adapter.

Integrates with Okra's Harvest cloud API for mesh-grid topology
monitoring and per-node metrics. Uses httpx with transport-level
retries for reliability over cellular connections.

Gap feature (Gen318_Valuable_Unknown_Truths.md, Truth 15):
Mesh-grid topology monitoring — track node-to-node connections,
power sharing flows, and cluster health for Okra deployments.

STATUS: the endpoint paths and payload fields below are ASSUMED, not taken
from Okra documentation. Okra issues private per-operator API keys on request
and has not published this API; until a real key and response sample exist,
treat this adapter as a scaffold, not a working connector.

Asset identity: every reading must carry a Gen318 asset UUID. The cloud
ingest refuses anything else and the sync engine's pre-flight marks a
non-UUID row rejected locally (``edge/sync/engine.py::_post_measurements``),
so a reading with an invented id is collected, stored and then thrown away.
The adapter therefore requires an explicit ``asset_mappings`` entry — the
hub for topology metrics, one entry per Okra node id for node metrics — and
SKIPS (with a once-per-id warning) anything unmapped or mapped to a
non-UUID. Nothing is synthesised from a vendor id.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

from .base import AdapterCapabilities, AdapterStatus, HardwareAdapter, Reading

logger = structlog.get_logger()


def _mapped_asset_id(value: Any) -> str | None:
    """Return ``value`` as a canonical UUID string, or None if it is not one."""
    if value is None:
        return None
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return None


class OkraHarvestAdapter(HardwareAdapter):
    """Polls Okra Solar Harvest API for mesh-grid data.

    Monitors topology health, per-node metrics, and power-sharing flows.
    Designed for intermittent connectivity — transport-level retries
    handle transient cellular failures.
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._config: dict[str, Any] = {}
        self._status = AdapterStatus.DISCONNECTED
        self._last_reading_at: float = 0.0
        self._topology: dict[str, Any] = {}
        # Vendor ids already warned about, so an unmapped node logs once per
        # process instead of once every poll on a 5-minute loop.
        self._warned_unmapped: set[str] = set()

    def _resolve(self, key: str, value: Any) -> str | None:
        """Resolve one mapping entry; warn once when it is missing or invalid."""
        resolved = _mapped_asset_id(value)
        if resolved is None and key not in self._warned_unmapped:
            self._warned_unmapped.add(key)
            logger.warning(
                "okra_asset_unmapped",
                vendor_id=key,
                configured=value is not None,
                hint="add a Gen318 asset UUID under asset_mappings; readings are skipped",
            )
        return resolved

    async def initialize(self, config: dict[str, Any]) -> bool:
        """Connect to Okra API and validate credentials."""
        self._config = config

        transport = httpx.AsyncHTTPTransport(retries=2)
        self._client = httpx.AsyncClient(
            base_url=config["api_url"],
            headers={"Authorization": f"Bearer {config['api_key']}"},
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0),
            transport=transport,
        )

        try:
            resp = await self._client.get("/sites")
            if resp.status_code == 200:
                self._status = AdapterStatus.CONNECTED
                logger.info(
                    "okra_initialized",
                    api_url=config["api_url"],
                    site_id=config.get("site_id"),
                )
                return True
        except httpx.ConnectError:
            logger.warning("okra_connect_failed", api_url=config["api_url"])

        self._status = AdapterStatus.ERROR
        return False

    async def poll(self) -> list[Reading]:
        """Pull mesh-grid data from Okra API."""
        if not self._client:
            return []

        readings: list[Reading] = []
        now = datetime.now(UTC)
        site_id = self._config["site_id"]
        asset_mappings = self._config.get("asset_mappings", {})
        hub_asset_id = self._resolve("hub", asset_mappings.get("hub"))

        try:
            # Get site topology
            resp = await self._client.get(f"/sites/{site_id}/topology")
            if resp.status_code == 200:
                topology = resp.json()
                self._topology = topology

            if resp.status_code == 200 and hub_asset_id is not None:
                readings.append(
                    Reading(
                        metric_name="mesh.node_count",
                        value=float(len(self._topology.get("nodes", []))),
                        timestamp=now,
                        quality=0,
                        raw_register="okra:topology:node_count",
                        asset_id=hub_asset_id,
                    )
                )
                readings.append(
                    Reading(
                        metric_name="mesh.active_connections",
                        value=float(len(self._topology.get("connections", []))),
                        timestamp=now,
                        quality=0,
                        raw_register="okra:topology:connections",
                        asset_id=hub_asset_id,
                    )
                )

            # Get per-node metrics
            resp = await self._client.get(f"/sites/{site_id}/nodes")
            if resp.status_code == 200:
                nodes = resp.json().get("nodes", [])
                node_mappings = asset_mappings.get("nodes", {})

                for node in nodes:
                    node_id = str(node["id"])
                    node_asset_id = self._resolve(
                        f"node:{node_id}", node_mappings.get(node_id)
                    )
                    if node_asset_id is None:
                        continue

                    if "solar_power_w" in node:
                        readings.append(
                            Reading(
                                metric_name="solar.pv_power_w",
                                value=float(node["solar_power_w"]),
                                timestamp=now,
                                quality=0,
                                raw_register=f"okra:node:{node_id}:solar",
                                asset_id=node_asset_id,
                            )
                        )

                    if "battery_soc_pct" in node:
                        readings.append(
                            Reading(
                                metric_name="battery.soc_pct",
                                value=float(node["battery_soc_pct"]),
                                timestamp=now,
                                quality=0,
                                raw_register=f"okra:node:{node_id}:soc",
                                asset_id=node_asset_id,
                            )
                        )

                    if "load_power_w" in node:
                        readings.append(
                            Reading(
                                metric_name="meter.power_w",
                                value=float(node["load_power_w"]),
                                timestamp=now,
                                quality=0,
                                raw_register=f"okra:node:{node_id}:load",
                                asset_id=node_asset_id,
                            )
                        )

                    if "power_sharing_w" in node:
                        readings.append(
                            Reading(
                                metric_name="mesh.power_sharing_w",
                                value=float(node["power_sharing_w"]),
                                timestamp=now,
                                quality=0,
                                raw_register=f"okra:node:{node_id}:sharing",
                                asset_id=node_asset_id,
                            )
                        )

            if readings:
                self._last_reading_at = time.time()
                self._status = AdapterStatus.CONNECTED
                logger.debug("okra_poll_complete", reading_count=len(readings))

        except httpx.HTTPError as exc:
            # Catches all httpx network errors: ConnectError, TimeoutException,
            # ReadError, RemoteProtocolError, etc. — common on 2G cellular links
            self._status = AdapterStatus.ERROR
            logger.warning(
                "okra_poll_error", error=str(type(exc).__name__), detail=str(exc)
            )

        return readings

    async def subscribe(
        self,
        callback: Callable[[list[Reading]], Awaitable[None]],
    ) -> bool:
        """Okra API is poll-based; wrap in timed loop."""
        interval = self._config.get("poll_interval_seconds", 300)
        while True:
            readings = await self.poll()
            if readings:
                await callback(readings)
            await asyncio.sleep(interval)

    async def write(self, metric_name: str, value: float) -> bool:
        """Okra adapter is read-only."""
        return False

    async def health_check(self) -> AdapterStatus:
        """Return ERROR if no readings in 10 minutes."""
        if not self._client:
            return AdapterStatus.DISCONNECTED
        if self._last_reading_at > 0 and time.time() - self._last_reading_at > 600:
            return AdapterStatus.ERROR
        return self._status

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            can_read=True,
            can_write=False,
            can_subscribe=False,
            supported_metrics=[
                "solar.pv_power_w",
                "battery.soc_pct",
                "meter.power_w",
                "mesh.node_count",
                "mesh.active_connections",
                "mesh.power_sharing_w",
            ],
            poll_interval_seconds=self._config.get("poll_interval_seconds", 300),
            min_poll_interval_seconds=60,
        )

    async def shutdown(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
        self._status = AdapterStatus.DISCONNECTED
        logger.info("okra_shutdown")
