"""Edge watchdog service — container health monitor + tamper detection (Doc 14 §8.7).

Runs as the gen318-watchdog container (privileged). Monitors the other
5 containers via Docker SDK, restarts unhealthy ones within 60 seconds,
and manages the GPIO tamper detector.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any

import docker
import psutil
import structlog

from common.config import EdgeConfig, load_config
from common.mqtt_client import (
    TOPIC_ALERTS_CRITICAL,
    TOPIC_WATCHDOG_HEALTH,
    TOPIC_WATCHDOG_RESTART,
    MQTTClient,
)
from watchdog.tamper import TamperDetector

logger = structlog.get_logger()

# Container name prefix
CONTAINER_PREFIX = "gen318-"


class WatchdogService:
    """Monitors container health and system vitals."""

    MONITORED_CONTAINERS = [
        "gen318-mqtt",
        "gen318-collector",
        "gen318-inference",
        "gen318-sync",
        "gen318-api",
    ]

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self.check_interval = config.check_interval
        self.mqtt = MQTTClient(
            host=config.mqtt_host,
            port=config.mqtt_port,
            client_id="gen318-watchdog",
        )
        self._restart_counts: dict[str, int] = {}
        self._running = False

        try:
            self.docker_client = docker.from_env()
        except docker.errors.DockerException:
            logger.warning(
                "docker_not_available",
                msg="Docker socket not accessible — container monitoring disabled",
            )
            self.docker_client = None

        self.tamper_detector = TamperDetector(
            alert_callback=self._send_tamper_alert,
            gpio_pin=config.tamper_gpio_pin,
        )

    async def run(self) -> None:
        """Main watchdog loop."""
        self.tamper_detector.setup()
        self.mqtt.connect()
        self._running = True

        logger.info(
            "watchdog_started",
            check_interval=self.check_interval,
            containers=self.MONITORED_CONTAINERS,
            tamper_armed=self.tamper_detector.is_armed,
        )

        while self._running:
            try:
                await self._check_containers()
                await self._report_system_health()
            except Exception:
                logger.exception("watchdog_cycle_error")

            await asyncio.sleep(self.check_interval)

    async def _check_containers(self) -> None:
        """Check all monitored containers, restart if unhealthy."""
        if self.docker_client is None:
            return

        for name in self.MONITORED_CONTAINERS:
            try:
                container = self.docker_client.containers.get(name)
                status = container.status
                health = container.attrs.get("State", {}).get("Health", {})
                health_status = health.get("Status", "none")

                if status != "running" or health_status == "unhealthy":
                    self._restart_counts[name] = self._restart_counts.get(name, 0) + 1

                    logger.warning(
                        "container_restart",
                        container=name,
                        status=status,
                        health=health_status,
                        restart_count=self._restart_counts[name],
                    )

                    container.restart(timeout=30)

                    self.mqtt.publish(
                        TOPIC_WATCHDOG_RESTART,
                        {
                            "container": name,
                            "reason": f"status={status}, health={health_status}",
                            "restart_count": self._restart_counts[name],
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                    )

            except docker.errors.NotFound:
                logger.error("container_not_found", container=name)
                await self._send_critical_alert(f"Container {name} not found")
            except docker.errors.APIError:
                logger.exception("container_check_error", container=name)

    async def _report_system_health(self) -> None:
        """Publish system metrics to local MQTT and write shared status file."""
        import os

        container_status = self.get_container_status()
        system_metrics = {
            "cpu_pct": psutil.cpu_percent(interval=1),
            "ram_pct": psutil.virtual_memory().percent,
            "disk_pct": (
                psutil.disk_usage("/data").percent
                if os.path.exists("/data")
                else psutil.disk_usage("/").percent
            ),
            "temp_c": self._read_cpu_temperature(),
            "uptime_s": int(time.time() - psutil.boot_time()),
        }

        # Publish to MQTT
        mqtt_payload = {
            **system_metrics,
            "timestamp": datetime.now(UTC).isoformat(),
            "container_restarts": dict(self._restart_counts),
        }
        self.mqtt.publish(TOPIC_WATCHDOG_HEALTH, mqtt_payload)

        # Write shared status file for sync engine to read
        self._write_status_file(container_status, system_metrics)

    def _write_status_file(
        self,
        container_status: dict[str, str],
        system_metrics: dict[str, Any],
    ) -> None:
        """Atomically write system status to shared volume for sync engine.

        Uses os.replace() for atomic write — sync container reads this
        file to populate heartbeat payloads with real container/system data.
        """
        import os

        status_dir = "/data/metrics"
        status_path = os.path.join(status_dir, "system.json")
        tmp_path = status_path + ".tmp"

        try:
            os.makedirs(status_dir, exist_ok=True)
            payload = {
                "container_status": container_status,
                "system_metrics": system_metrics,
                "updated_at": time.time(),
            }
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, status_path)
        except OSError:
            logger.warning("status_file_write_failed", path=status_path)

    @staticmethod
    def _read_cpu_temperature() -> float:
        """Read SoC die temperature from sysfs."""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return int(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError):
            return -1.0

    async def _send_tamper_alert(self, alert_data: dict[str, Any]) -> None:
        """Handle tamper detection — send P0 alert via MQTT QoS 2."""
        self.mqtt.publish(TOPIC_ALERTS_CRITICAL, alert_data, qos=2)
        logger.critical(
            "tamper_alert_sent",
            alert=alert_data,
        )

    async def _send_critical_alert(self, message: str) -> None:
        """Send a critical watchdog alert."""
        self.mqtt.publish(
            TOPIC_ALERTS_CRITICAL,
            {
                "type": "watchdog_alert",
                "severity": "critical",
                "message": message,
                "device_id": self.config.device_id,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            qos=2,
        )

    def stop(self) -> None:
        self._running = False
        self.tamper_detector.cleanup()
        self.mqtt.disconnect()
        logger.info("watchdog_stopped")

    def get_container_status(self) -> dict[str, str]:
        """Get current status of all monitored containers."""
        if self.docker_client is None:
            return {name: "unknown" for name in self.MONITORED_CONTAINERS}

        statuses: dict[str, str] = {}
        for name in self.MONITORED_CONTAINERS:
            try:
                container = self.docker_client.containers.get(name)
                statuses[name] = container.status
            except docker.errors.NotFound:
                statuses[name] = "not_found"
            except docker.errors.APIError:
                statuses[name] = "error"
        return statuses


async def main() -> None:
    config = load_config()
    service = WatchdogService(config)
    try:
        await service.run()
    except KeyboardInterrupt:
        service.stop()


if __name__ == "__main__":
    asyncio.run(main())
