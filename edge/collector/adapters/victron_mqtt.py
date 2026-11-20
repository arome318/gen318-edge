"""Victron Energy MQTT adapter for real-time bidirectional data (Doc 15 §7.2, F-HW-02).

Subscribes to a Victron GX device's local MQTT broker for real-time
telemetry. The GX device (Cerbo GX, Venus GX) publishes data with
topic format: {prefix}/{portalId}/{service}/{instance}/{path}

Prefixes: N/ = notify, R/ = read request, W/ = write

Uses aiomqtt with the officially recommended reconnection pattern
(while/try/except MqttError). Write commands are queued and dispatched
inside the subscription loop since aiomqtt only allows publishing
within the ``async with client`` context.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import structlog

from .base import AdapterCapabilities, AdapterStatus, HardwareAdapter, Reading

logger = structlog.get_logger()


# ── MQTT Topic Maps ──────────────────────────────────────────

# System (com.victronenergy.system, instance 0)
MQTT_TOPICS_SYSTEM: dict[str, str] = {
    "battery.soc_pct": "N/{portal}/system/0/Dc/Battery/Soc",
    "battery.voltage_v": "N/{portal}/system/0/Dc/Battery/Voltage",
    "battery.current_a": "N/{portal}/system/0/Dc/Battery/Current",
    "battery.power_w": "N/{portal}/system/0/Dc/Battery/Power",
    "battery.temperature_c": "N/{portal}/system/0/Dc/Battery/Temperature",
    "battery.consumed_ah": "N/{portal}/system/0/Dc/Battery/ConsumedAmphours",
    "battery.time_to_go_s": "N/{portal}/system/0/Dc/Battery/TimeToGo",
    "solar.pv_power_w": "N/{portal}/system/0/Dc/Pv/Power",
    "solar.pv_current_a": "N/{portal}/system/0/Dc/Pv/Current",
    "system.ac_consumption_w": "N/{portal}/system/0/Ac/Consumption/L1/Power",
    "system.grid_power_w": "N/{portal}/system/0/Ac/Grid/L1/Power",
    "genset.power_w": "N/{portal}/system/0/Ac/Genset/L1/Power",
}

# Solar Charger (com.victronenergy.solarcharger)
MQTT_TOPICS_SOLAR: dict[str, str] = {
    "solar.pv_voltage_v": "N/{portal}/solarcharger/{inst}/Pv/V",
    "solar.pv_current_a": "N/{portal}/solarcharger/{inst}/Pv/I",
    "solar.pv_power_w": "N/{portal}/solarcharger/{inst}/Yield/Power",
    "solar.daily_yield_kwh": "N/{portal}/solarcharger/{inst}/History/Daily/0/Yield",
    "solar.max_power_today_w": "N/{portal}/solarcharger/{inst}/History/Daily/0/MaxPower",
    "solar.charger_state": "N/{portal}/solarcharger/{inst}/State",
    "solar.total_yield_kwh": "N/{portal}/solarcharger/{inst}/Yield/User",
}

# VE.Bus / Multi / Quattro (com.victronenergy.vebus)
MQTT_TOPICS_VEBUS: dict[str, str] = {
    "inverter.ac_input_voltage_v": "N/{portal}/vebus/{inst}/Ac/ActiveIn/L1/V",
    "inverter.ac_input_current_a": "N/{portal}/vebus/{inst}/Ac/ActiveIn/L1/I",
    "inverter.ac_input_freq_hz": "N/{portal}/vebus/{inst}/Ac/ActiveIn/L1/F",
    "inverter.ac_input_power_w": "N/{portal}/vebus/{inst}/Ac/ActiveIn/L1/P",
    "inverter.ac_output_voltage_v": "N/{portal}/vebus/{inst}/Ac/Out/L1/V",
    "inverter.ac_output_current_a": "N/{portal}/vebus/{inst}/Ac/Out/L1/I",
    "inverter.ac_output_freq_hz": "N/{portal}/vebus/{inst}/Ac/Out/L1/F",
    "inverter.ac_output_power_w": "N/{portal}/vebus/{inst}/Ac/Out/L1/P",
    "inverter.state": "N/{portal}/vebus/{inst}/State",
    "inverter.error_code": "N/{portal}/vebus/{inst}/VebusError",
}

# Battery Monitor (com.victronenergy.battery)
MQTT_TOPICS_BATTERY: dict[str, str] = {
    "battery.cell_min_voltage_v": "N/{portal}/battery/{inst}/System/MinCellVoltage",
    "battery.cell_max_voltage_v": "N/{portal}/battery/{inst}/System/MaxCellVoltage",
    "battery.cell_min_temp_c": "N/{portal}/battery/{inst}/System/MinCellTemperature",
    "battery.cell_max_temp_c": "N/{portal}/battery/{inst}/System/MaxCellTemperature",
    "battery.soh_pct": "N/{portal}/battery/{inst}/Soh",
    "battery.capacity_ah": "N/{portal}/battery/{inst}/Capacity",
}


class VictronMQTTAdapter(HardwareAdapter):
    """Subscribes to Victron GX MQTT topics for real-time telemetry.

    Uses aiomqtt's recommended reconnection pattern. Write commands
    are queued and dispatched within the subscription loop.
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._portal_id: str = ""
        self._broker_host: str = ""
        self._broker_port: int = 1883
        self._topic_to_metric: dict[
            str, tuple[str, str]
        ] = {}  # topic -> (metric, asset_id)
        self._status = AdapterStatus.DISCONNECTED
        self._last_reading_at: float = 0.0
        self._write_queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue()
        self._write_topics: dict[str, str] = {}

    async def initialize(self, config: dict[str, Any]) -> bool:
        """Build topic maps from config. Connection happens in subscribe()."""
        self._config = config
        self._portal_id = config["portal_id"]
        self._broker_host = config["broker_host"]
        self._broker_port = config.get("broker_port", 1883)
        self._build_topic_map()
        raw_write_topics = config.get("write_topics") or {}
        if isinstance(raw_write_topics, dict):
            self._write_topics = {
                str(metric): str(topic).format(portal=self._portal_id)
                for metric, topic in raw_write_topics.items()
                if isinstance(metric, str) and isinstance(topic, str)
            }
        self._status = AdapterStatus.CONNECTING

        logger.info(
            "victron_mqtt_initialized",
            broker=self._broker_host,
            portal_id=self._portal_id,
            topic_count=len(self._topic_to_metric),
        )
        return True

    def _build_topic_map(self) -> None:
        """Build reverse map from MQTT topic string to (metric_name, asset_id)."""
        portal = self._portal_id
        mappings = self._config.get("asset_mappings", {})

        # System topics
        for metric, topic_template in MQTT_TOPICS_SYSTEM.items():
            topic = topic_template.format(portal=portal)
            asset_id = mappings.get("system", "unknown")
            self._topic_to_metric[topic] = (metric, asset_id)

        # Solar charger topics (per instance)
        for inst in self._config.get("instances", {}).get("solarcharger", [0]):
            for metric, topic_template in MQTT_TOPICS_SOLAR.items():
                topic = topic_template.format(portal=portal, inst=inst)
                asset_id = mappings.get(f"solarcharger_{inst}", "unknown")
                self._topic_to_metric[topic] = (metric, asset_id)

        # VE.Bus topics (per instance)
        for inst in self._config.get("instances", {}).get("vebus", [0]):
            for metric, topic_template in MQTT_TOPICS_VEBUS.items():
                topic = topic_template.format(portal=portal, inst=inst)
                asset_id = mappings.get(f"vebus_{inst}", "unknown")
                self._topic_to_metric[topic] = (metric, asset_id)

        # Battery monitor topics (per instance)
        for inst in self._config.get("instances", {}).get("battery", [0]):
            for metric, topic_template in MQTT_TOPICS_BATTERY.items():
                topic = topic_template.format(portal=portal, inst=inst)
                asset_id = mappings.get(f"battery_{inst}", "unknown")
                self._topic_to_metric[topic] = (metric, asset_id)

    async def poll(self) -> list[Reading]:
        """Not used for MQTT — use subscribe() instead."""
        return []

    async def subscribe(
        self,
        callback: Callable[[list[Reading]], Awaitable[None]],
    ) -> bool:
        """Subscribe to all mapped MQTT topics with auto-reconnect.

        Uses aiomqtt's recommended pattern: reconnection is handled by
        re-entering the ``async with`` context manager after MqttError.
        Write commands from the queue are dispatched inside the loop.
        """
        import aiomqtt

        backoff = 5.0
        max_backoff = 60.0

        while True:
            try:
                client = aiomqtt.Client(
                    hostname=self._broker_host,
                    port=self._broker_port,
                    keepalive=60,
                )
                async with client:
                    self._status = AdapterStatus.CONNECTED
                    backoff = 5.0  # reset on successful connect

                    # Subscribe to all topics
                    for topic in self._topic_to_metric:
                        await client.subscribe(topic)

                    # Request current values via R/ prefix
                    for topic in self._topic_to_metric:
                        read_topic = topic.replace("N/", "R/", 1)
                        await client.publish(read_topic, "")

                    logger.info(
                        "victron_mqtt_subscribed",
                        topic_count=len(self._topic_to_metric),
                    )

                    async for message in client.messages:
                        # Process incoming message
                        topic_str = str(message.topic)
                        if topic_str in self._topic_to_metric:
                            reading = self._parse_message(topic_str, message.payload)
                            if reading is not None:
                                self._last_reading_at = time.time()
                                await callback([reading])

                        # Dispatch any pending write commands
                        while not self._write_queue.empty():
                            metric_name, value = self._write_queue.get_nowait()
                            await self._dispatch_write(client, metric_name, value)

            except aiomqtt.MqttError as exc:
                self._status = AdapterStatus.ERROR
                logger.warning(
                    "victron_mqtt_disconnected",
                    error=str(exc),
                    reconnect_in=backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    def _parse_message(
        self, topic_str: str, payload: bytes | bytearray
    ) -> Reading | None:
        """Parse a Victron MQTT message into a Reading."""
        metric_name, asset_id = self._topic_to_metric[topic_str]

        try:
            decoded = json.loads(payload)
            if isinstance(decoded, dict):
                raw_val = decoded.get("value")
                if raw_val is None:
                    return None  # device offline — do not emit
                value = float(raw_val)
            else:
                value = float(decoded)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

        return Reading(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(UTC),
            quality=0,
            raw_register=topic_str,
            asset_id=asset_id,
        )

    async def _dispatch_write(
        self, client: Any, metric_name: str, value: float
    ) -> None:
        """Publish only to an installer-approved write topic."""
        write_topic = self._write_topics.get(metric_name)
        if write_topic:
            await client.publish(write_topic, json.dumps({"value": value}))
            logger.info(
                "victron_mqtt_write",
                metric=metric_name,
                value=value,
                topic=write_topic,
            )

    async def write(self, metric_name: str, value: float) -> bool:
        """Queue a write command for dispatch inside the subscription loop.

        Returns True if the command was queued. Actual delivery happens
        asynchronously when the subscription loop processes the queue.
        """
        if metric_name not in self._write_topics:
            return False
        self._write_queue.put_nowait((metric_name, value))
        return True

    async def health_check(self) -> AdapterStatus:
        """Return ERROR if no reading in 2 minutes."""
        if self._status == AdapterStatus.DISCONNECTED:
            return AdapterStatus.DISCONNECTED
        if self._last_reading_at > 0 and time.time() - self._last_reading_at > 120:
            return AdapterStatus.ERROR
        return self._status

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            can_read=True,
            can_write=bool(self._write_topics),
            can_subscribe=True,
            supported_metrics=[m for m, _ in self._topic_to_metric.values()],
            supported_write_metrics=sorted(self._write_topics),
            poll_interval_seconds=0,  # real-time via subscription
            min_poll_interval_seconds=0,
        )

    async def shutdown(self) -> None:
        """Mark as disconnected. aiomqtt cleanup is handled by context manager."""
        self._status = AdapterStatus.DISCONNECTED
        logger.info("victron_mqtt_shutdown")
