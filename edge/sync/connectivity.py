"""Hybrid connectivity detection using modem + latency probes (Doc 14 §8.4).

Primary: mmcli (ModemManager) for instant access technology / RSSI
detection with zero data cost. Fallback: HTTP latency probe every
5th cycle for environments without ModemManager.
"""

from __future__ import annotations

import asyncio
import json
import time

import structlog

from sync.engine import ConnectivityState

logger = structlog.get_logger()

# Access technology → ConnectivityState mapping
ACCESS_TECH_MAP: dict[str, ConnectivityState] = {
    "lte": ConnectivityState.FOURG,
    "lte-a": ConnectivityState.FOURG,
    "lte-advanced": ConnectivityState.FOURG,
    "umts": ConnectivityState.THREEG,
    "hspa": ConnectivityState.THREEG,
    "hspa+": ConnectivityState.THREEG,
    "hsdpa": ConnectivityState.THREEG,
    "hsupa": ConnectivityState.THREEG,
    "gsm": ConnectivityState.TWOG,
    "gprs": ConnectivityState.TWOG,
    "edge": ConnectivityState.TWOG,
}


class ConnectivityDetector:
    """Hybrid connectivity detector: modem-first, latency-fallback."""

    def __init__(self, modem_index: int = 0) -> None:
        self._modem_index = modem_index
        self._cycle_count = 0
        self._last_state = ConnectivityState.OFFLINE
        self._modem_available: bool | None = None
        self._last_detection_time: float = 0

    async def detect(self) -> ConnectivityState:
        """Detect current connectivity state."""
        self._cycle_count += 1

        # Re-probe modem availability every 50 cycles (USB re-enumeration recovery)
        if self._modem_available is False and self._cycle_count % 50 == 0:
            self._modem_available = None

        # Try modem detection first (zero data cost)
        if self._modem_available is not False:
            state = await self._detect_via_modem()
            if state is not None:
                self._modem_available = True
                self._last_state = state
                self._last_detection_time = time.time()
                return state
            else:
                self._modem_available = False
                logger.info(
                    "modem_not_available",
                    msg="Falling back to latency-based detection",
                )

        # Fallback: latency probe (every 5th cycle to save data)
        if self._cycle_count % 5 == 1:
            state = await self._detect_via_latency()
            self._last_state = state
            self._last_detection_time = time.time()
            return state

        return self._last_state

    async def _detect_via_modem(self) -> ConnectivityState | None:
        """Use mmcli (ModemManager) to read access technology and signal."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "mmcli",
                "-m",
                str(self._modem_index),
                "--output-json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode != 0:
                return None

            data = json.loads(stdout.decode())
            modem = data.get("modem", {})

            # Extract access technology
            generic = modem.get("generic", {})
            access_tech_list = generic.get("access-technologies", [])

            if not access_tech_list:
                return ConnectivityState.OFFLINE

            # Use the best available access technology
            best_state = ConnectivityState.OFFLINE
            for tech in access_tech_list:
                tech_lower = tech.lower().strip()
                mapped = ACCESS_TECH_MAP.get(tech_lower)
                if mapped and mapped > best_state:
                    best_state = mapped

            logger.debug(
                "modem_detection",
                technologies=access_tech_list,
                state=best_state.name,
            )
            return best_state

        except FileNotFoundError:
            # mmcli not installed
            return None
        except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
            return None

    async def _detect_via_latency(
        self, cloud_url: str = "https://api.gen318.com"
    ) -> ConnectivityState:
        """Fallback: detect connectivity via HTTP latency probe."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                start = time.monotonic()
                resp = await client.get(f"{cloud_url}/health")
                latency_ms = (time.monotonic() - start) * 1000

                if resp.status_code != 200:
                    return ConnectivityState.OFFLINE

                if latency_ms < 200:
                    return ConnectivityState.FOURG
                elif latency_ms < 800:
                    return ConnectivityState.THREEG
                elif latency_ms < 3000:
                    return ConnectivityState.TWOG
                else:
                    return ConnectivityState.OFFLINE

        except (httpx.ConnectError, httpx.TimeoutException):
            return ConnectivityState.OFFLINE

    @property
    def last_state(self) -> ConnectivityState:
        return self._last_state

    @property
    def seconds_since_detection(self) -> float:
        if self._last_detection_time == 0:
            return float("inf")
        return time.time() - self._last_detection_time
