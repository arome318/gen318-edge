"""Rule-based fault detection on the box.

Pure functions over the latest readings. No model files and no network are
needed, so these rules work on a brand-new box with nothing downloaded. The
XGBoost models in ``inference/`` add to these when model files are present.

Every threshold is configurable per site (``config/sms.yaml``). The defaults
are conservative starting points for LFP battery systems and 230 V AC output,
not certified limits: tune them with the site's installer.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, fields
from typing import Any


@dataclass
class Thresholds:
    comms_lost_s: int = 600
    fresh_s: int = 900
    soc_warning_pct: float = 20.0
    soc_critical_pct: float = 10.0
    battery_temp_c: float = 45.0
    cell_spread_v: float = 0.15
    inverter_temp_c: float = 70.0
    overload_ratio: float = 0.9
    outage_voltage_v: float = 150.0
    genset_soc_pct: float = 60.0
    genset_min_power_w: float = 200.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Thresholds:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass(frozen=True)
class Fault:
    code: str
    severity: str  # "critical" | "warning"
    asset_id: str
    values: dict[str, float]

    @property
    def key(self) -> str:
        return f"{self.code}:{self.asset_id}"


class SiteState:
    """Latest value and time of every metric, per asset. Thread-safe."""

    def __init__(self) -> None:
        self._latest: dict[str, dict[str, tuple[float, float]]] = {}
        self._lock = threading.Lock()
        self.last_seen_s: float | None = None

    def update(self, asset_id: str, metric: str, value: float, ts_s: float) -> None:
        with self._lock:
            self._latest.setdefault(asset_id, {})[metric] = (float(value), ts_s)
            if self.last_seen_s is None or ts_s > self.last_seen_s:
                self.last_seen_s = ts_s

    def get(self, asset_id: str, metric: str, now_s: float, max_age_s: float) -> float | None:
        with self._lock:
            entry = self._latest.get(asset_id, {}).get(metric)
        if entry is None or now_s - entry[1] > max_age_s:
            return None
        return entry[0]

    def assets(self) -> list[str]:
        with self._lock:
            return sorted(self._latest)

    def find(self, metric: str, now_s: float, max_age_s: float) -> list[tuple[str, float]]:
        out = []
        for asset in self.assets():
            value = self.get(asset, metric, now_s, max_age_s)
            if value is not None:
                out.append((asset, value))
        return out


def evaluate(
    state: SiteState,
    t: Thresholds,
    now_s: float,
    rated_power_w: dict[str, float] | None = None,
) -> list[Fault]:
    """Return every fault condition true right now (before debouncing)."""
    faults: list[Fault] = []
    rated_power_w = rated_power_w or {}
    age = t.fresh_s

    if state.last_seen_s is not None and now_s - state.last_seen_s > t.comms_lost_s:
        faults.append(
            Fault("comms_lost", "critical", "site",
                  {"minutes": round((now_s - state.last_seen_s) / 60)})
        )
        return faults  # stale data: other rules would be guesses

    for asset in state.assets():
        g = lambda m, a=asset: state.get(a, m, now_s, age)  # noqa: E731

        soc = g("battery.soc_pct")
        if soc is not None:
            if soc < t.soc_critical_pct:
                faults.append(Fault("battery_critical", "critical", asset, {"soc": round(soc)}))
            elif soc < t.soc_warning_pct:
                faults.append(Fault("battery_low", "warning", asset, {"soc": round(soc)}))

        temps = [v for v in (g("battery.temperature_c"), g("battery.cell_max_temp_c")) if v is not None]
        if temps and max(temps) > t.battery_temp_c:
            faults.append(Fault("battery_hot", "critical", asset, {"temp": round(max(temps))}))

        vmax, vmin = g("battery.cell_max_voltage_v"), g("battery.cell_min_voltage_v")
        if vmax is not None and vmin is not None and vmax - vmin > t.cell_spread_v:
            faults.append(Fault("cell_imbalance", "warning", asset,
                                {"spread_mv": round((vmax - vmin) * 1000)}))

        code = g("inverter.error_code")
        if code is not None and code != 0:
            faults.append(Fault("inverter_fault", "critical", asset, {"error_code": int(code)}))

        inv_t = g("inverter.temperature_c")
        if inv_t is not None and inv_t > t.inverter_temp_c:
            faults.append(Fault("inverter_hot", "warning", asset, {"temp": round(inv_t)}))

        out_w = g("inverter.ac_output_power_w")
        if out_w is None:
            out_w = g("load.power_w")
        rated = rated_power_w.get(asset)
        if out_w is not None and rated and out_w > t.overload_ratio * rated:
            faults.append(Fault("overload", "warning", asset,
                                {"load_pct": round(100 * out_w / rated)}))

        out_v = g("inverter.ac_output_voltage_v")
        if out_v is not None and out_v < t.outage_voltage_v:
            faults.append(Fault("power_off", "critical", asset, {"voltage": round(out_v)}))

        gen_w = g("genset.power_w")
        if gen_w is not None and soc is not None and gen_w > t.genset_min_power_w and soc > t.genset_soc_pct:
            faults.append(Fault("diesel_waste", "warning", asset, {"soc": round(soc)}))

    return faults


class FaultTracker:
    """Debounce faults: alert only after ``hold_s``, re-alert after ``realert_s``."""

    def __init__(self, hold_s: int = 120, realert_s: int = 6 * 3600) -> None:
        self.hold_s = hold_s
        self.realert_s = realert_s
        self._first_seen: dict[str, float] = {}
        self._active: dict[str, tuple[Fault, float]] = {}

    @property
    def active(self) -> dict[str, Fault]:
        return {k: f for k, (f, _) in self._active.items()}

    def step(self, faults: list[Fault], now_s: float) -> tuple[list[Fault], list[Fault]]:
        """Return (faults to alert now, faults that just cleared)."""
        present = {f.key: f for f in faults}
        to_alert: list[Fault] = []
        for key, fault in present.items():
            if key in self._active:
                _, alerted = self._active[key]
                if now_s - alerted >= self.realert_s:
                    self._active[key] = (fault, now_s)
                    to_alert.append(fault)
                else:
                    self._active[key] = (fault, alerted)
                continue
            first = self._first_seen.setdefault(key, now_s)
            if now_s - first >= self.hold_s:
                self._active[key] = (fault, now_s)
                self._first_seen.pop(key, None)
                to_alert.append(fault)
        for key in list(self._first_seen):
            if key not in present:
                del self._first_seen[key]
        cleared = [f for k, (f, _) in self._active.items() if k not in present]
        for f in cleared:
            del self._active[f.key]
        return to_alert, cleared
