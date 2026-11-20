"""MQTT client wrapper for edge-side inter-container communication (Doc 14).

Wraps paho-mqtt with structured logging, exponential backoff reconnection,
and typed topic constants for the Gen318 edge MQTT namespace.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import paho.mqtt.client as mqtt
import structlog

logger = structlog.get_logger()

# ── Topic Constants ──────────────────────────────────────────────
TOPIC_TELEMETRY = "gen318/telemetry"
TOPIC_EVENTS = "gen318/events"
TOPIC_ALERTS_CRITICAL = "gen318/alerts/critical"
TOPIC_WATCHDOG_HEALTH = "gen318/watchdog/health"
TOPIC_WATCHDOG_RESTART = "gen318/watchdog/restart"
TOPIC_CONTROL_SSH = "gen318/control/ssh"
TOPIC_INFERENCE_RESULTS = "gen318/inference/results"
TOPIC_SITELEARN = "gen318/sitelearn/findings"
TOPIC_MODEL_UPDATES = "gen318/model/updates"
TOPIC_CONTROL_DIAGNOSTICS = "gen318/control/diagnostics"
TOPIC_CONTROL_ADAPTER_RECONNECT = "gen318/control/adapter/reconnect"


class MQTTClient:
    """Thread-safe MQTT client with automatic reconnection."""

    def __init__(
        self,
        host: str = "mosquitto",
        port: int = 1883,
        client_id: str = "gen318-edge",
    ) -> None:
        self._host = host
        self._port = port
        self._client = mqtt.Client(
            client_id=client_id,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self._connected = False
        self._subscriptions: list[tuple[str, Callable, int]] = []

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        rc: int | mqtt.ReasonCode,
        properties: Any = None,
    ) -> None:
        """Handle successful connection — resubscribe to topics."""
        self._connected = True
        logger.info("mqtt_connected", host=self._host, port=self._port)

        # Resubscribe on reconnect
        for topic, _callback, qos in self._subscriptions:
            client.subscribe(topic, qos=qos)
            logger.info("mqtt_resubscribed", topic=topic)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any = None,
        rc: int | mqtt.ReasonCode = 0,
        properties: Any = None,
    ) -> None:
        """Handle disconnection."""
        self._connected = False
        if rc != 0:
            logger.warning("mqtt_unexpected_disconnect", rc=rc)

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        msg: mqtt.MQTTMessage,
    ) -> None:
        """Route incoming messages to registered callbacks."""
        for topic, callback, _qos in self._subscriptions:
            if mqtt.topic_matches_sub(topic, msg.topic):
                try:
                    payload = json.loads(msg.payload.decode())
                    callback(payload)
                except json.JSONDecodeError:
                    logger.warning(
                        "mqtt_invalid_json",
                        topic=msg.topic,
                        payload_size=len(msg.payload),
                    )
                except Exception:
                    logger.exception("mqtt_callback_error", topic=msg.topic)

    def connect(self, max_retries: int = 0) -> None:
        """Connect to the MQTT broker with exponential backoff.

        Args:
            max_retries: Maximum retry attempts. 0 = infinite.
        """
        attempt = 0
        backoff = 5.0  # seconds
        max_backoff = 60.0

        while True:
            try:
                self._client.connect(self._host, self._port, keepalive=60)
                self._client.loop_start()
                logger.info("mqtt_connecting", host=self._host, port=self._port)
                return
            except (ConnectionRefusedError, OSError) as exc:
                attempt += 1
                if max_retries and attempt >= max_retries:
                    raise RuntimeError(
                        f"Failed to connect to MQTT after {attempt} attempts"
                    ) from exc

                logger.warning(
                    "mqtt_connect_retry",
                    attempt=attempt,
                    backoff_s=backoff,
                    error=str(exc),
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    def publish(
        self,
        topic: str,
        payload: dict[str, Any] | str,
        qos: int = 0,
    ) -> None:
        """Publish a JSON message to a topic."""
        if isinstance(payload, dict):
            data = json.dumps(payload).encode()
        else:
            data = payload.encode()

        result = self._client.publish(topic, data, qos=qos)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.warning(
                "mqtt_publish_failed",
                topic=topic,
                rc=result.rc,
            )

    def subscribe(
        self,
        topic: str,
        callback: Callable[[dict[str, Any]], None],
        qos: int = 0,
    ) -> None:
        """Subscribe to a topic with a callback for JSON payloads."""
        self._subscriptions.append((topic, callback, qos))
        if self._connected:
            self._client.subscribe(topic, qos=qos)
        logger.info("mqtt_subscribed", topic=topic, qos=qos)

    def disconnect(self) -> None:
        """Disconnect from the broker."""
        self._client.loop_stop()
        self._client.disconnect()
        self._connected = False
        logger.info("mqtt_disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected
