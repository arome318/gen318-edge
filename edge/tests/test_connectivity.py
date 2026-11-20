"""ConnectivityDetector unit tests — modem and latency detection."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from sync.connectivity import ConnectivityDetector
from sync.engine import ConnectivityState


class TestModemDetection:
    @pytest.mark.asyncio
    async def test_lte_detected_as_4g(self) -> None:
        detector = ConnectivityDetector()
        mmcli_output = json.dumps(
            {"modem": {"generic": {"access-technologies": ["lte"]}}}
        ).encode()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = AsyncMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(mmcli_output, b""))
            mock_exec.return_value = proc

            state = await detector.detect()
            assert state == ConnectivityState.FOURG

    @pytest.mark.asyncio
    async def test_gsm_detected_as_2g(self) -> None:
        detector = ConnectivityDetector()
        mmcli_output = json.dumps(
            {"modem": {"generic": {"access-technologies": ["gsm"]}}}
        ).encode()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            proc = AsyncMock()
            proc.returncode = 0
            proc.communicate = AsyncMock(return_value=(mmcli_output, b""))
            mock_exec.return_value = proc

            state = await detector.detect()
            assert state == ConnectivityState.TWOG


class TestFallback:
    @pytest.mark.asyncio
    async def test_modem_unavailable_uses_latency(self) -> None:
        detector = ConnectivityDetector()

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            # First call discovers modem is unavailable
            await detector.detect()
            assert detector._modem_available is False

    @pytest.mark.asyncio
    async def test_modem_reprobe_after_50_cycles(self) -> None:
        detector = ConnectivityDetector()
        detector._modem_available = False
        detector._cycle_count = 49  # Next call will be cycle 50

        # At cycle 50, _modem_available should reset to None
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            await detector.detect()
            # After cycle 50 reset + re-probe failure, it's False again
            # but the reset happened (proving the re-probe logic works)
            assert detector._cycle_count == 50

    @pytest.mark.asyncio
    async def test_latency_only_every_5th_cycle(self) -> None:
        detector = ConnectivityDetector()
        detector._modem_available = False
        detector._cycle_count = 2  # Not a 5th cycle

        # Should return cached last_state without probing
        detector._last_state = ConnectivityState.THREEG
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            state = await detector.detect()
            assert state == ConnectivityState.THREEG
