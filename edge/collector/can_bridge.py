"""Standalone CAN BMS reader for sites with a CAN tap.

A SocketCAN interface lives in a network namespace. The collector container
has its own, so a `can0` on the host is invisible to it — and no Docker
feature hands a container ONE host interface while keeping its own namespace
(macvlan and ipvlan are Ethernet-level and do not apply to CAN). The only
options are to share the host's whole namespace or to leave the container
model behind.

So this exists: a small process that runs in the collector's image with host
networking, reads the bus, and writes measurements to the same SQLite file
the collector writes. That confines host networking to one single-purpose
process of a few hundred lines, instead of granting it to the collector —
which loads every vendor adapter and its credentials and is the largest
surface in the stack.

Writing straight to SQLite rather than republishing over MQTT is deliberate.
`inference` is the only subscriber to the telemetry topic and nothing writes
that topic to storage; the collector persists from its own adapters. A
bridge that only published would reach local inference and never leave the
node. Two containers already write this database concurrently — the
collector inserts measurements and sync updates their synced flags, over a
write-ahead journal — so this is the pattern the stack already runs, not a
new one.

Offline-first throughout. It never contacts the cloud, a broker outage
cannot stop a reading from being stored, and a refused bus is a logged exit
rather than a crash loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from typing import Any

import structlog

from collector.adapters.base import Reading
from collector.adapters.can_bms_listener import CANBMSListenerAdapter
from common.database import EdgeDatabase
from common.mqtt_client import TOPIC_TELEMETRY, MQTTClient

logger = structlog.get_logger()

#: How long to wait before retrying a bus that would not open. A refused
#: interface is usually a commissioning problem that a human has to fix, so
#: retrying fast only fills the log.
_RETRY_SECONDS = 60.0


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


def adapter_config() -> dict[str, Any]:
    """The listener's configuration, from the environment.

    Deliberately read here rather than added to ``EdgeConfig``: this process
    is opt-in and its settings are meaningless to the other five containers.
    """
    return {
        "interface": os.environ.get("EDGE_CAN_INTERFACE", "can0"),
        "bitrate": int(os.environ.get("EDGE_CAN_BITRATE", "500000")),
        "profile": os.environ.get("EDGE_CAN_PROFILE", "victron_generic"),
        "asset_id": os.environ.get("EDGE_CAN_ASSET_ID", ""),
        # The listen-only gate. Only an isolated bench bus should ever set
        # this false, and it has to be written down to happen.
        "require_listen_only": _env_bool("EDGE_CAN_REQUIRE_LISTEN_ONLY", True),
    }


class CANBridge:
    """Reads a Victron-profile BMS bus and persists what it hears."""

    def __init__(
        self,
        db: EdgeDatabase,
        config: dict[str, Any],
        *,
        device_id: str = "",
        mqtt: MQTTClient | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self.db = db
        self._config = config
        self._device_id = device_id
        self._mqtt = mqtt
        self._interval = max(0.1, poll_interval_seconds)
        self._running = False
        self._adapter: CANBMSListenerAdapter | None = None

    async def run(self) -> None:
        """Poll until stopped. Never raises."""
        self._running = True
        while self._running:
            adapter = CANBMSListenerAdapter()
            started = False
            try:
                started = await adapter.initialize(self._config)
            except Exception:
                logger.exception("can_bridge_initialize_failed")
            if not started:
                # The adapter has already logged WHY — a bus that is not
                # listen-only, or an image that cannot check. Both need a
                # person, so wait rather than hammering.
                logger.warning(
                    "can_bridge_not_started",
                    retry_in_seconds=_RETRY_SECONDS,
                    interface=self._config.get("interface"),
                )
                await self._sleep(_RETRY_SECONDS)
                continue

            self._adapter = adapter
            logger.info(
                "can_bridge_started",
                interface=self._config.get("interface"),
                profile=self._config.get("profile"),
                identity=adapter.identity(),
            )
            try:
                await self._poll_forever(adapter)
            finally:
                self._adapter = None
                try:
                    await adapter.shutdown()
                except Exception:
                    logger.exception("can_bridge_shutdown_failed")

    async def _poll_forever(self, adapter: CANBMSListenerAdapter) -> None:
        while self._running:
            try:
                readings = await adapter.poll()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("can_bridge_poll_failed")
                await self._sleep(self._interval)
                continue
            if readings:
                self._store(readings)
            await self._sleep(self._interval)

    def _store(self, readings: list[Reading]) -> None:
        """Persist first, publish second.

        SQLite is the durable path and the only one that reaches the cloud;
        MQTT feeds local inference. Doing the write first means a broker
        problem can never cost a reading.
        """
        batch = [
            (
                int(reading.timestamp.timestamp() * 1000),
                reading.asset_id or "unknown",
                reading.metric_name,
                reading.value,
                reading.quality,
            )
            for reading in readings
        ]
        try:
            stored = self.db.insert_measurements(batch)
            logger.debug("can_bridge_stored", count=stored)
        except Exception:
            logger.exception("can_bridge_store_failed", count=len(batch))
            return

        if self._mqtt is None:
            return
        for reading in readings:
            try:
                self._mqtt.publish(
                    TOPIC_TELEMETRY,
                    {
                        "timestamp": int(reading.timestamp.timestamp() * 1000),
                        "asset_id": reading.asset_id,
                        "metric_name": reading.metric_name,
                        "value": reading.value,
                        "quality": reading.quality,
                        "device_id": self._device_id,
                    },
                )
            except Exception:
                # Best effort only: the reading is already durable.
                logger.debug("can_bridge_publish_failed", metric=reading.metric_name)

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            self._running = False
            raise

    def stop(self) -> None:
        self._running = False


async def main() -> int:  # pragma: no cover - process entrypoint
    config = adapter_config()
    if not config["asset_id"]:
        logger.error(
            "can_bridge_missing_asset_id",
            msg="EDGE_CAN_ASSET_ID must name the battery asset this bus reports",
        )
        return 2

    db = EdgeDatabase(
        os.environ.get("SQLITE_PATH", "/data/gen318.db"),
        os.environ.get("SQLITE_ENCRYPTION_KEY", ""),
    )
    db.connect()
    db.init_schema()

    mqtt: MQTTClient | None = None
    if _env_bool("EDGE_CAN_PUBLISH_MQTT", True):
        try:
            mqtt = MQTTClient(
                host=os.environ.get("MQTT_HOST", "localhost"),
                port=int(os.environ.get("MQTT_PORT", "1883")),
                client_id="gen318-can-bridge",
            )
            mqtt.connect()
        except Exception:
            # Inference loses a live feed; storage and sync are unaffected.
            logger.warning("can_bridge_mqtt_unavailable", exc_info=True)
            mqtt = None

    bridge = CANBridge(
        db=db,
        config=config,
        device_id=os.environ.get("DEVICE_ID", ""),
        mqtt=mqtt,
        poll_interval_seconds=float(os.environ.get("EDGE_CAN_POLL_SECONDS", "1")),
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, bridge.stop)

    try:
        await bridge.run()
    finally:
        if mqtt is not None:
            try:
                mqtt.disconnect()
            except Exception:
                logger.debug("can_bridge_mqtt_disconnect_failed")
        db.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    raise SystemExit(asyncio.run(main()))
