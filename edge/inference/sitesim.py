"""Simulated solar-battery mini-grid, with optional injected faults.

Used to test on-site learning without hardware. It is a simplified physical
model (clear-sky solar with day-to-day cloud, a 48 V LFP bank with internal
resistance, an evening-peak village load), not a validated site model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

STEP_S = 300
DAY_S = 86400
T0 = 1_772_323_200.0  # 1 Mar 2026 00:00 UTC


@dataclass
class Faults:
    soiling_day: float | None = None      # panels lose output from this day
    soiling_factor: float = 0.6
    resistance_day: float | None = None   # battery internal resistance rises
    resistance_mult: float = 2.0
    theft_day: float | None = None        # extra night load from this day
    theft_w: float = 800.0


@dataclass
class SimSite:
    days: int = 60
    seed: int = 0
    faults: Faults = field(default_factory=Faults)
    pv_kw: float = 6.0
    bank_kwh: float = 15.0
    utc_offset_h: float = 1.0

    def run(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(self.seed)
        n = int(self.days * DAY_S / STEP_S)
        ts = T0 + np.arange(n) * STEP_S
        local = ts + self.utc_offset_h * 3600
        hour = (local % DAY_S) / 3600
        day = np.floor((ts - T0) / DAY_S)
        dow = (np.floor(local / DAY_S) + 3) % 7

        clear = np.clip(np.sin(np.pi * (hour - 6.5) / 12), 0, None) ** 1.3
        day_cloud = rng.beta(5, 2, size=self.days + 1)
        dips = rng.random(n) > day_cloud[day.astype(int)]
        cloud = np.where(dips, rng.uniform(0.2, 0.8, n), 1.0)
        pv = self.pv_kw * 1000 * clear * cloud * rng.normal(1, 0.03, n)
        f = self.faults
        if f.soiling_day is not None:
            pv = np.where(day >= f.soiling_day, pv * f.soiling_factor, pv)
        pv = np.clip(pv, 0, None)

        load = 350 + 900 * ((hour >= 7) & (hour < 18)) + 1800 * ((hour >= 18) & (hour < 23))
        load = load * np.where(dow >= 5, 1.1, 1.0) * rng.normal(1, 0.12, n)
        if f.theft_day is not None:
            load = load + np.where((day >= f.theft_day) & (hour < 5), f.theft_w, 0.0)
        load = np.clip(load, 50, None)

        soc = np.empty(n)
        volt = np.empty(n)
        bpow = np.empty(n)  # Victron convention: positive = charging
        outv = np.empty(n)
        temp = 30 + 5 * np.sin(np.pi * (hour - 9) / 12) + rng.normal(0, 0.3, n)
        s, off = 70.0, False
        cap_wh = self.bank_kwh * 1000
        for k in range(n):
            r = 0.02 * (f.resistance_mult if f.resistance_day is not None and day[k] >= f.resistance_day else 1.0)
            if off and s > 20:
                off = False
            if s <= 5:
                off = True
            served = 0.0 if off else load[k]
            net = pv[k] - served  # + charges
            if s >= 99 and net > 0:
                net = 0.0
            s = float(np.clip(s + net * STEP_S / 3600 / cap_wh * 100, 0, 100))
            ocv = 50.0 + 3.0 * s / 100 - 2.5 * np.exp(-s / 6) + 1.5 * np.exp((s - 100) / 3)
            current = net / ocv
            volt[k] = ocv + current * r + rng.normal(0, 0.03)
            soc[k], bpow[k] = s, net
            outv[k] = 0.0 if off else 230 + rng.normal(0, 1.5)
        return {
            "solar.pv_power_w": (ts, pv),
            "system.ac_consumption_w": (ts, np.where(outv > 0, load, 0.0)),
            "battery.voltage_v": (ts, volt),
            "battery.soc_pct": (ts, soc),
            "battery.power_w": (ts, bpow),
            "battery.temperature_c": (ts, temp),
            "inverter.ac_output_voltage_v": (ts, outv),
        }


class SimHistory:
    """A read-only history over simulated data, with a movable 'now'."""

    def __init__(self, data: dict[str, tuple[np.ndarray, np.ndarray]], asset: str = "inv1") -> None:
        self.data = data
        self.asset = asset
        self.now = float(max(ts[-1] for ts, _ in data.values()))

    def get_series(self, asset: str, metric: str, hours: int) -> tuple[np.ndarray, np.ndarray]:
        if asset != self.asset or metric not in self.data:
            return np.array([]), np.array([])
        ts, v = self.data[metric]
        lo = np.searchsorted(ts, self.now - hours * 3600)
        hi = np.searchsorted(ts, self.now, side="right")
        return ts[lo:hi], v[lo:hi]

    def get_readings_history(self, asset: str, metric: str, hours: int = 24) -> list[dict[str, Any]]:
        ts, v = self.get_series(asset, metric, hours)
        return [{"timestamp": t * 1000, "value": x, "quality": 0} for t, x in zip(ts, v, strict=True)]

    def get_latest_readings(self) -> list[dict[str, Any]]:
        return [{"asset_id": self.asset, "metric_name": m, "value": 0.0, "quality": 0,
                 "timestamp": self.now * 1000} for m in self.data]
