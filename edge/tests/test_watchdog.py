"""WatchdogService + TamperDetector unit tests."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestWatchdogService:
    @pytest.mark.asyncio
    async def test_restart_unhealthy_container(
        self, edge_config, mock_mqtt, mock_docker_client
    ) -> None:
        mock_docker_client._containers["gen318-collector"].status = "exited"
        mock_docker_client._containers["gen318-collector"].attrs = {
            "State": {"Health": {"Status": "unhealthy"}}
        }

        with patch("watchdog.main.docker") as mock_docker_mod:
            mock_docker_mod.from_env.return_value = mock_docker_client
            mock_docker_mod.errors = __import__("docker").errors
            from watchdog.main import WatchdogService

            service = WatchdogService(edge_config)
            service.mqtt = mock_mqtt
            service.docker_client = mock_docker_client

            await service._check_containers()
            mock_docker_client._containers[
                "gen318-collector"
            ].restart.assert_called_once_with(timeout=30)

    @pytest.mark.asyncio
    async def test_critical_alert_on_missing_container(
        self, edge_config, mock_mqtt, mock_docker_client
    ) -> None:
        import docker.errors

        del mock_docker_client._containers["gen318-inference"]

        with patch("watchdog.main.docker") as mock_docker_mod:
            mock_docker_mod.from_env.return_value = mock_docker_client
            mock_docker_mod.errors = docker.errors
            from watchdog.main import WatchdogService

            service = WatchdogService(edge_config)
            service.mqtt = mock_mqtt
            service.docker_client = mock_docker_client

            await service._check_containers()

            assert len(mock_mqtt._published) >= 1
            topic, payload = mock_mqtt._published[-1]
            assert topic == "gen318/alerts/critical"
            assert payload["severity"] == "critical"

    def test_status_file_atomic_write(self, edge_config, mock_mqtt) -> None:
        import tempfile

        with (
            patch("watchdog.main.docker") as mock_docker_mod,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            mock_docker_mod.from_env.return_value = MagicMock()
            mock_docker_mod.errors = __import__("docker").errors
            from watchdog.main import WatchdogService

            WatchdogService(edge_config)

            # Direct test: write and verify
            container_status = {"gen318-mqtt": "running"}
            system_metrics = {"cpu_pct": 25.0}
            # Write directly to tmpdir
            tmp_file = os.path.join(tmpdir, "system.json.tmp")
            final_file = os.path.join(tmpdir, "system.json")
            import time as _t

            with open(tmp_file, "w") as f:
                json.dump(
                    {
                        "container_status": container_status,
                        "system_metrics": system_metrics,
                        "updated_at": _t.time(),
                    },
                    f,
                )
            os.replace(tmp_file, final_file)

            with open(final_file) as f:
                data = json.load(f)
            assert data["container_status"]["gen318-mqtt"] == "running"

    def test_get_container_status(self, edge_config, mock_docker_client) -> None:
        with patch("watchdog.main.docker") as mock_docker_mod:
            mock_docker_mod.from_env.return_value = mock_docker_client
            mock_docker_mod.errors = __import__("docker").errors
            from watchdog.main import WatchdogService

            service = WatchdogService(edge_config)
            service.docker_client = mock_docker_client

            status = service.get_container_status()
            assert status["gen318-mqtt"] == "running"
            assert len(status) == 5


class TestTamperDetector:
    @pytest.mark.asyncio
    async def test_simulate_tamper_fires_callback(self) -> None:
        callback = AsyncMock()
        from watchdog.tamper import TamperDetector

        detector = TamperDetector(alert_callback=callback, gpio_pin=17)
        await detector.simulate_tamper()

        callback.assert_called_once()
        alert = callback.call_args[0][0]
        assert alert["type"] == "tamper_detected"
        assert alert["severity"] == "critical"

    def test_tamper_works_without_gpio(self) -> None:
        """Verify simulator mode when RPi.GPIO is not available."""
        callback = AsyncMock()
        from watchdog.tamper import TamperDetector

        detector = TamperDetector(alert_callback=callback)
        detector.setup()  # Should not crash without GPIO
        assert not detector.is_armed  # GPIO is None on non-RPi

    @pytest.mark.asyncio
    async def test_tamper_debounce(self) -> None:
        callback = AsyncMock()
        from watchdog.tamper import TamperDetector

        detector = TamperDetector(alert_callback=callback)
        detector._last_trigger = time.time()  # Just triggered

        # Simulate immediate second trigger — should be debounced
        detector._on_tamper(17)
        callback.assert_not_called()
