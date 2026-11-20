"""GPIO-based tamper detection for edge node enclosure.

Monitors BCM pin 17 (configurable) connected to the IP65 enclosure
lid switch. Falling edge triggers an immediate P0 critical alert
via MQTT QoS 2, delivered within 1 second of enclosure open.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger()

# Try importing RPi.GPIO — falls back to None for simulator mode
try:
    import RPi.GPIO as GPIO  # type: ignore[import-untyped]
except ImportError:
    GPIO = None


class TamperDetector:
    """Detects physical tampering via GPIO interrupt on enclosure switch."""

    def __init__(
        self,
        alert_callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
        gpio_pin: int = 17,
    ) -> None:
        self.alert_callback = alert_callback
        self._gpio_pin = gpio_pin
        self._armed = False
        self._last_trigger: float = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    def setup(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Initialize GPIO pin for tamper detection."""
        if GPIO is None:
            logger.info(
                "tamper_simulator_mode",
                msg="RPi.GPIO not available — tamper detection in simulator mode",
            )
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self._loop = loop or asyncio.get_running_loop()
        GPIO.add_event_detect(
            self._gpio_pin,
            GPIO.FALLING,
            callback=self._on_tamper,
            bouncetime=200,
        )
        self._armed = True
        logger.info(
            "tamper_detection_armed",
            gpio_pin=self._gpio_pin,
        )

    def _on_tamper(self, channel: int) -> None:
        """GPIO interrupt handler — enclosure opened.

        Called from GPIO thread. Schedules async alert callback
        on the event loop for immediate P0 delivery.
        """
        now = time.time()

        # Extra debounce: ignore triggers within 5 seconds
        if now - self._last_trigger < 5.0:
            return
        self._last_trigger = now

        logger.critical(
            "tamper_detected",
            gpio_pin=channel,
            timestamp=now,
        )

        if self._loop is None:
            logger.error("tamper_alert_no_loop", gpio_pin=channel)
            return

        alert_data = {
            "type": "tamper_detected",
            "severity": "critical",
            "source": "gpio",
            "pin": channel,
            "timestamp": int(now * 1000),
        }
        self._loop.call_soon_threadsafe(
            self._loop.create_task,
            self.alert_callback(alert_data),
        )

    async def simulate_tamper(self) -> None:
        """Simulate a tamper event for testing (no hardware required)."""
        logger.info("tamper_simulated")
        await self.alert_callback(
            {
                "type": "tamper_detected",
                "severity": "critical",
                "source": "simulator",
                "pin": self._gpio_pin,
                "timestamp": int(time.time() * 1000),
            }
        )

    def cleanup(self) -> None:
        """Release GPIO resources."""
        if GPIO is not None and self._armed:
            GPIO.cleanup(self._gpio_pin)
            self._armed = False
            logger.info("tamper_detection_disarmed")

    @property
    def is_armed(self) -> bool:
        return self._armed
