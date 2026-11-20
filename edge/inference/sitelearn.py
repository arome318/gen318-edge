"""On-site learning: each box learns what normal looks like at its own site.

No external data, no cloud and no pre-trained weights. After a site has
enough history (default 7 days for solar and load, 14 days for the battery),
the box trains three small XGBoost models on that site's own readings:

* **Solar**: the clear-sky potential of each time of day (90th-percentile
  quantile regression). Panels that keep producing well below their own
  learned potential for days point to dust (Harmattan), shading, or a string
  or MPPT fault. Clouds lower individual days; the rule needs several days.
* **Battery**: battery voltage as a function of state of charge, power and
  temperature, learned once as a baseline. Ageing cells and loose or corroded
  terminals both show as voltage sagging further under load than the
  baseline predicts. The baseline is frozen so slow drift is not absorbed;
  reset it after a battery replacement.
* **Load**: the normal band of site load by time of day and day of week
  (10th, 50th and 90th percentile). A sustained load far above the band at
  odd hours can mean an illegal connection or a fault on a feeder; far below
  it with power on can mean a tripped feeder.

* **Run-out**: the battery's usual net discharge by time of day, and how
  many watt-hours one percent of charge is worth (both learned from the
  site's own readings). Walking that forward from the present state of
  charge gives the time the bank is likely to reach its cut-off, so the
  technician hears before the lights go out, not after.

Each model is a few hundred kilobytes, trains in seconds on a Raspberry Pi
and never leaves the box. Findings are published as warnings for the SMS
service; nothing here acts on equipment.

Model files and ``meta.json`` are written to a temporary file and renamed,
because these boxes lose power without warning; a damaged file is dropped
and relearned instead of stopping the service.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import structlog

logger = structlog.get_logger()

SLOT_S = 900  # 15-minute slots
DAY_S = 86400

PV_METRIC = "solar.pv_power_w"
# Inverters without a total PV register report each tracker separately.
MPPT_METRICS = tuple(f"solar.mppt{i}_power_w" for i in range(1, 5))
LOAD_METRICS = ("system.ac_consumption_w", "load.power_w", "inverter.ac_output_power_w")
BATTERY_V = "battery.voltage_v"
BATTERY_SOC = "battery.soc_pct"
BATTERY_P = "battery.power_w"
BATTERY_I = "battery.current_a"
BATTERY_T = "battery.temperature_c"
OUTPUT_V = "inverter.ac_output_voltage_v"


class History(Protocol):
    def get_readings_history(self, asset_id: str, metric: str, hours: int = 24) -> list[dict[str, Any]]: ...
    def get_latest_readings(self) -> list[dict[str, Any]]: ...


@dataclass
class SiteLearnConfig:
    utc_offset_hours: float = 1.0
    pv_min_days: int = 7
    pv_window_days: int = 30
    pv_retrain_days: int = 30
    pv_alert_days: int = 5
    pv_alert_ratio: float = 0.75
    battery_min_days: int = 14
    battery_alert_days: int = 7
    battery_alert_pct: float = 0.6
    load_min_days: int = 7
    load_window_days: int = 28
    load_retrain_hours: int = 24
    load_alert_slots: int = 8
    load_alert_hits: int = 6
    load_high_factor: float = 1.3
    load_high_margin_w: float = 300.0
    load_low_factor: float = 0.4
    load_low_min_expected_w: float = 300.0
    # Off-grid inverters throttle the panels once the bank is full, so those
    # slots say nothing about panel health: leave them out of the solar model.
    pv_full_soc_pct: float = 95.0
    pv_strong_hours: tuple[float, float] = (9.0, 15.0)
    pv_min_slots_per_day: int = 6
    runout_min_days: int = 7
    runout_window_days: int = 28
    runout_retrain_hours: int = 24
    runout_soc_pct: float = 10.0       # where the inverter cuts the output
    runout_horizon_hours: float = 12.0
    runout_alert_hours: float = 6.0    # only warn when it is this close
    runout_min_soc_swing_pct: float = 100.0
    # A bank that empties most nights is a sizing problem the technician
    # already knows about; a nightly SMS would only teach them to ignore it.
    runout_routine_days: int = 4       # of the last 7
    # The forecast must pass the cut-off by this much before anyone is woken:
    # a night predicted to end a point or two above empty is a coin toss.
    runout_margin_pct: float = 5.0


# Technician replies of "false alarm" loosen a learned warning by this much per
# step (at most TUNING_MAX_STEPS). See sms/tuning.py for how steps are decided.
TUNING_MAX_STEPS = 2


def tuned_config(base: SiteLearnConfig, steps: dict[str, int]) -> SiteLearnConfig:
    """Return ``base`` with each learned warning loosened by its step count."""
    def n(code: str) -> int:
        try:
            return max(0, min(TUNING_MAX_STEPS, int(steps.get(code, 0))))
        except (TypeError, ValueError):
            return 0

    return replace(
        base,
        pv_alert_ratio=base.pv_alert_ratio - 0.05 * n("pv_underperform"),
        battery_alert_pct=base.battery_alert_pct * (1 + 0.25 * n("battery_degrading")),
        load_high_factor=base.load_high_factor + 0.15 * n("load_high"),
        load_low_factor=max(0.1, base.load_low_factor - 0.05 * n("load_low")),
        runout_alert_hours=max(2.0, base.runout_alert_hours - 1.5 * n("battery_runout")),
    )


def atomic_write(path: Path, data: bytes) -> None:
    """Write then rename, so a power cut leaves the old file or the new one."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ── series helpers ─────────────────────────────────────────────


def slot_series(rows: list[dict[str, Any]], slot_s: int = SLOT_S) -> tuple[np.ndarray, np.ndarray]:
    """Average readings into fixed time slots. Returns (slot start seconds, mean value)."""
    if not rows:
        return np.array([]), np.array([])
    ts = np.array([float(r["timestamp"]) for r in rows]) / 1000.0
    vals = np.array([float(r["value"]) for r in rows])
    quality = np.array([int(r.get("quality", 0)) for r in rows])
    keep = np.isfinite(vals) & (quality <= 2)
    if not keep.any():
        return np.array([]), np.array([])
    slots = np.floor(ts[keep] / slot_s) * slot_s
    uniq, inv = np.unique(slots, return_inverse=True)
    sums = np.bincount(inv, weights=vals[keep])
    counts = np.bincount(inv)
    return uniq, sums / counts


def align(*series: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Keep only the slots present in every series."""
    common = series[0][0]
    for ts, _ in series[1:]:
        common = np.intersect1d(common, ts)
    out = []
    for ts, vals in series:
        idx = np.searchsorted(ts, common)
        out.append(vals[idx])
    return common, out


def time_features(ts_s: np.ndarray, utc_offset_h: float) -> np.ndarray:
    """Columns: hour sin, hour cos, weekday sin, weekday cos, weekend flag (local time)."""
    local = ts_s + utc_offset_h * 3600
    hour = (local % DAY_S) / 3600.0
    days = np.floor(local / DAY_S)
    dow = (days + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        (dow >= 5).astype(float),
    ])


def local_hour(ts_s: np.ndarray, utc_offset_h: float) -> np.ndarray:
    return ((ts_s + utc_offset_h * 3600) % DAY_S) / 3600.0


def local_day(ts_s: np.ndarray, utc_offset_h: float) -> np.ndarray:
    return np.floor((ts_s + utc_offset_h * 3600) / DAY_S)


# ── models ─────────────────────────────────────────────────────


def _xgb() -> Any:
    import xgboost as xgb

    return xgb


def fit_quantile(X: np.ndarray, y: np.ndarray, alpha: float, rounds: int = 120) -> Any:
    xgb = _xgb()
    params = {"objective": "reg:quantileerror", "quantile_alpha": alpha, "max_depth": 4,
              "eta": 0.1, "min_child_weight": 3, "subsample": 0.9, "nthread": 2, "verbosity": 0}
    return xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=rounds)


def fit_regression(X: np.ndarray, y: np.ndarray, rounds: int = 200) -> Any:
    xgb = _xgb()
    params = {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.08,
              "min_child_weight": 5, "subsample": 0.9, "nthread": 2, "verbosity": 0}
    return xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=rounds)


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    return model.predict(_xgb().DMatrix(X))


def pv_daily_peak_ratio(ts: np.ndarray, actual: np.ndarray, expected: np.ndarray,
                        utc_offset_h: float, hours: tuple[float, float] = (9.0, 15.0),
                        min_slots: int = 6) -> dict[int, float]:
    """Per local day: 90th percentile of actual/expected over strong-sun slots.

    Clouds pull most slots down but rarely the best ones; dust, shading or a
    dead string caps every slot, so this ratio separates the two.
    """
    if len(ts) == 0:
        return {}
    hour = local_hour(ts, utc_offset_h)
    strong = ((hour >= hours[0]) & (hour <= hours[1])
              & (expected > 0.25 * max(float(np.max(expected)), 1.0)))
    days = local_day(ts, utc_offset_h)
    out: dict[int, float] = {}
    for d in np.unique(days[strong]):
        sel = strong & (days == d)
        if sel.sum() >= min_slots:
            out[int(d)] = float(np.percentile(actual[sel] / np.maximum(expected[sel], 1.0), 90))
    return out


class SiteLearner:
    """Trains and applies the three on-site models for every asset."""

    def __init__(self, history: History, model_dir: str | Path,
                 config: SiteLearnConfig | None = None) -> None:
        self.db = history
        self.base_cfg = config or SiteLearnConfig()
        self.cfg = self.base_cfg
        self.tuning: dict[str, int] = {}
        self.dir = Path(model_dir) / "site_learning"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"
        self.meta: dict[str, Any] = self._read_meta()
        self._models: dict[str, Any] = {}

    def set_tuning(self, steps: dict[str, Any]) -> None:
        """Apply sensitivity steps decided from technician labels (SMS service)."""
        clean = {str(k): int(v) for k, v in (steps or {}).items()
                 if isinstance(v, (int, float)) and int(v) > 0}
        if clean != self.tuning:
            logger.info("site_learning_tuning", steps=clean)
        self.tuning = clean
        self.cfg = tuned_config(self.base_cfg, clean)

    # ── persistence ──
    def _read_meta(self) -> dict[str, Any]:
        try:
            data = json.loads(self.meta_path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            # Truncated by a power cut, or a failing SD card. Start again:
            # the models retrain from the readings still on the box.
            logger.warning("site_learning_meta_unreadable", path=str(self.meta_path))
            return {}
        return data if isinstance(data, dict) else {}

    def _save_meta(self) -> None:
        atomic_write(self.meta_path, json.dumps(self.meta, indent=2, sort_keys=True).encode())

    def _model_file(self, key: str) -> Path:
        return self.dir / f"{key.replace(':', '__')}.ubj"

    def _store(self, key: str, model: Any, info: dict[str, Any]) -> None:
        raw = bytes(model.save_raw("ubj"))
        atomic_write(self._model_file(key), raw)
        self._models[key] = model
        self.meta[key] = {**info, "bytes": len(raw)}
        self._save_meta()

    def _load(self, key: str) -> Any | None:
        if key in self._models:
            return self._models[key]
        path = self._model_file(key)
        if not path.exists():
            return None
        try:
            booster = _xgb().Booster()
            booster.load_model(str(path))
        except Exception:
            logger.warning("site_learning_model_unreadable", key=key)
            return None
        self._models[key] = booster
        return booster

    def _forget(self, key: str) -> None:
        """Drop a model whose file is missing or damaged so it is learned again."""
        for k in [k for k in self.meta if k == key or k.startswith(key + ":")]:
            self.meta.pop(k)
            self._models.pop(k, None)
        self._save_meta()

    def reset(self, key_prefix: str) -> None:
        """Forget a baseline, e.g. 'battery:<asset>' after a battery replacement."""
        for key in [k for k in self.meta if k.startswith(key_prefix)]:
            self.meta.pop(key)
            self._models.pop(key, None)
        self._save_meta()

    def status(self) -> dict[str, Any]:
        return {k: {kk: v for kk, v in info.items() if kk != "model"} for k, info in self.meta.items()}

    # ── data ──
    def _assets_with(self, metric: str) -> list[str]:
        return sorted({r["asset_id"] for r in self.db.get_latest_readings() if r["metric_name"] == metric})

    def _series(self, asset: str, metric: str, days: int) -> tuple[np.ndarray, np.ndarray]:
        fast = getattr(self.db, "get_series", None)
        if fast is not None:
            ts, vals = fast(asset, metric, days * 24)
            if len(ts) == 0:
                return np.array([]), np.array([])
            slots = np.floor(ts / SLOT_S) * SLOT_S
            uniq, inv = np.unique(slots, return_inverse=True)
            return uniq, np.bincount(inv, weights=vals) / np.bincount(inv)
        return slot_series(self.db.get_readings_history(asset, metric, hours=days * 24))

    @staticmethod
    def _span_days(ts: np.ndarray) -> float:
        return float((ts[-1] - ts[0]) / DAY_S) if len(ts) > 1 else 0.0

    # ── solar ──
    def _pv_series(self, asset: str, days: int) -> tuple[np.ndarray, np.ndarray]:
        ts, pv = self._series(asset, PV_METRIC, days)
        if len(ts):
            return ts, pv
        parts = [self._series(asset, m, days) for m in MPPT_METRICS]
        parts = [p for p in parts if len(p[0])]
        if not parts:
            return np.array([]), np.array([])
        if len(parts) == 1:
            return parts[0]
        common, vals = align(*parts)
        return common, np.sum(vals, axis=0)

    def _pv_assets(self) -> list[str]:
        found: set[str] = set()
        for m in (PV_METRIC, *MPPT_METRICS):
            found |= set(self._assets_with(m))
        return sorted(found)

    def _uncurtailed(self, asset: str, ts: np.ndarray, days: int) -> np.ndarray:
        """True for slots where the panels were free to give all they could.

        With the bank full, an off-grid inverter throttles the panels down to
        the load, so low output then means "nothing to charge", not dust. The
        state of charge comes from the same asset, or from the site's battery
        when the inverter does not report one. With no state of charge at all
        every slot is kept, as before.
        """
        soc_ts, soc = self._series(asset, BATTERY_SOC, days)
        if len(soc_ts) == 0:
            others = [a for a in self._assets_with(BATTERY_SOC) if a != asset]
            if others:
                soc_ts, soc = self._series(others[0], BATTERY_SOC, days)
        free = np.ones(len(ts), bool)
        if len(soc_ts) == 0:
            return free
        idx = np.clip(np.searchsorted(soc_ts, ts), 0, len(soc_ts) - 1)
        known = soc_ts[idx] == ts
        free[known] = soc[idx[known]] < self.cfg.pv_full_soc_pct
        return free

    def _pv_ratios(self, ts: np.ndarray, pv: np.ndarray, model: Any) -> dict[int, float]:
        c = self.cfg
        exp = predict(model, time_features(ts, c.utc_offset_hours)[:, :2]) if len(ts) else np.array([])
        return pv_daily_peak_ratio(ts, pv, exp, c.utc_offset_hours, c.pv_strong_hours,
                                   c.pv_min_slots_per_day)

    def _pv(self, asset: str, now: float) -> dict[str, Any] | None:
        c = self.cfg
        key = f"pv:{asset}"
        ts, pv = self._pv_series(asset, c.pv_window_days)
        if len(ts) == 0:
            return None
        free = self._uncurtailed(asset, ts, c.pv_window_days)
        span_days = self._span_days(ts)
        last_ts = float(ts[-1])
        ts, pv = ts[free], pv[free]
        if len(ts) == 0:
            return None
        info = self.meta.get(key)
        if info is not None and self._load(key) is None:  # file lost or damaged: learn again
            self._forget(key)
            info = None
        due = info is None or now - info["trained_at"] >= c.pv_retrain_days * DAY_S
        if due and span_days >= c.pv_min_days:
            train = ts < (last_ts - c.pv_alert_days * DAY_S) if info else np.ones(len(ts), bool)
            if info:  # skip days the current model already flagged as weak
                ratios = self._pv_ratios(ts, pv, self._load(key))
                bad = {d for d, r in ratios.items() if r < 0.85 * info["baseline_ratio"]}
                train &= ~np.isin(local_day(ts, c.utc_offset_hours), list(bad))
            if train.sum() >= 100:
                X = time_features(ts[train], c.utc_offset_hours)[:, :2]
                started = time.perf_counter()
                model = fit_quantile(X, pv[train], 0.9)
                fit_s = time.perf_counter() - started
                ratios = self._pv_ratios(ts[train], pv[train], model)
                baseline = float(np.median(list(ratios.values()))) if ratios else 1.0
                self._store(key, model, {"trained_at": now, "rows": int(train.sum()),
                                         "days": round(self._span_days(ts[train]), 1),
                                         "curtailed_pct": round(100 * (1 - float(free.mean()))),
                                         "baseline_ratio": baseline, "kind": "solar",
                                         "fit_s": round(fit_s, 3)})
                info = self.meta[key]
        model = self._load(key)
        if model is None or info is None:
            return None
        recent = ts >= last_ts - c.pv_alert_days * DAY_S
        ratios = self._pv_ratios(ts[recent], pv[recent], model)
        # Days with too few free slots (bank full by mid-morning) give no
        # ratio, so a site that is curtailed all week is simply not judged.
        if len(ratios) < c.pv_alert_days - 1:
            return None
        rel = float(np.median(list(ratios.values()))) / max(info["baseline_ratio"], 1e-6)
        if rel < c.pv_alert_ratio:
            return {"code": "pv_underperform", "asset_id": asset, "severity": "warning",
                    "values": {"pct": round(100 * rel), "days": len(ratios)}}
        return None

    # ── battery ──
    def _battery_power(self, asset: str, days: int) -> tuple[np.ndarray, np.ndarray]:
        """Battery power in watts; from current times voltage if not reported."""
        p = self._series(asset, BATTERY_P, days)
        if len(p[0]):
            return p
        i = self._series(asset, BATTERY_I, days)
        v = self._series(asset, BATTERY_V, days)
        if len(i[0]) == 0 or len(v[0]) == 0:
            return np.array([]), np.array([])
        ts_i, (iv, vv) = align(i, v)
        return ts_i, iv * vv

    def _battery_frame(self, asset: str, days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        v = self._series(asset, BATTERY_V, days)
        soc = self._series(asset, BATTERY_SOC, days)
        p = self._battery_power(asset, days)
        if min(len(v[0]), len(soc[0]), len(p[0])) == 0:
            return None
        temp = self._series(asset, BATTERY_T, days)
        if len(temp[0]):
            ts, (vv, ss, pp, tt) = align(v, soc, p, temp)
        else:
            ts, (vv, ss, pp) = align(v, soc, p)
            tt = np.full(len(ts), np.nan)
        X = np.column_stack([ss, pp / 1000.0, np.abs(pp) / 1000.0, tt])
        return ts, X, vv

    def _battery(self, asset: str, now: float) -> dict[str, Any] | None:
        c = self.cfg
        key = f"battery:{asset}"
        info = self.meta.get(key)
        if info is None:
            frame = self._battery_frame(asset, c.battery_min_days + 2)
            if frame is None or self._span_days(frame[0]) < c.battery_min_days:
                return None
            ts, X, v = frame
            split = ts < ts[0] + (ts[-1] - ts[0]) * 0.8
            if split.sum() < 200 or (~split).sum() < 50:
                return None
            started = time.perf_counter()
            holdout_model = fit_regression(X[split], v[split])
            hold_resid = predict(holdout_model, X[~split]) - v[~split]
            model = fit_regression(X, v)
            fit_s = time.perf_counter() - started
            # Vendors disagree on the sign of battery power, so learn it: the
            # discharge direction is the one that moves state of charge down.
            dsoc = np.diff(X[:, 0])
            corr = np.corrcoef(X[1:, 1], dsoc)[0, 1] if np.std(dsoc) > 0 else 0.0
            sign = -1.0 if corr > 0 else 1.0
            discharge_kw_all = sign * X[:, 1]
            dis = discharge_kw_all > 0.05
            discharge_kw = discharge_kw_all[dis]
            soc_dis = X[dis, 0] if dis.any() else X[:, 0]
            mad = float(np.median(np.abs(hold_resid - np.median(hold_resid)))) * 1.4826
            self._store(key, model, {
                "trained_at": now, "rows": int(len(ts)), "days": round(self._span_days(ts), 1),
                "nominal_v": float(np.median(v)), "holdout_bias_v": float(np.median(hold_resid)),
                "holdout_mad_v": mad, "discharge_sign": sign,
                "load_kw_p50": float(np.median(discharge_kw)) if len(discharge_kw) else 0.0,
                "load_kw_p95": float(np.percentile(discharge_kw, 95)) if len(discharge_kw) else 0.0,
                # only judge the battery where the baseline has seen it operate
                "soc_lo": float(np.percentile(soc_dis, 10)), "soc_hi": float(np.percentile(soc_dis, 90)),
                "kind": "battery", "fit_s": round(fit_s, 3),
            })
            return None
        model = self._load(key)
        if model is None:  # file lost or damaged: take a new baseline
            self._forget(key)
            return None
        frame = self._battery_frame(asset, c.battery_alert_days)
        if frame is None:
            return None
        ts, X, v = frame
        kw = info["discharge_sign"] * X[:, 1]
        sel = ((kw > 0.05) & (kw <= info["load_kw_p95"])
               & (X[:, 0] >= info["soc_lo"]) & (X[:, 0] <= info["soc_hi"]))
        if sel.sum() < 60 or np.std(kw[sel]) < 1e-3:
            return None
        resid = predict(model, X[sel]) - v[sel]
        # Higher internal resistance (ageing cells, loose or corroded terminals)
        # adds a voltage drop PROPORTIONAL TO LOAD. Fitting the extra drop
        # against load separates that from a constant offset, which is what
        # model error in an unfamiliar state-of-charge range looks like.
        # State of charge enters as a covariate so a model error that depends
        # on SoC (e.g. more time at low charge after a cloudy week) is not
        # mistaken for a load-proportional sag.
        soc = X[sel, 0] / 100.0
        A = np.column_stack([np.ones(sel.sum()), kw[sel], soc, soc ** 2])
        coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
        dof = max(len(resid) - A.shape[1], 1)
        sigma2 = float(np.sum((resid - A @ coef) ** 2) / dof)
        cov = sigma2 * np.linalg.pinv(A.T @ A)
        slope, se = float(coef[1]), float(np.sqrt(max(cov[1, 1], 0.0)))  # volts per kW
        drop = slope * info["load_kw_p95"]  # extra sag at heavy (95th percentile) load
        threshold = c.battery_alert_pct / 100 * info["nominal_v"]
        if drop > threshold and slope > 5 * se:
            return {"code": "battery_degrading", "asset_id": asset, "severity": "warning",
                    "values": {"drop_pct": round(100 * drop / info["nominal_v"], 1)}}
        return None

    # ── load ──
    def _load_metric(self, asset: str) -> str | None:
        have = {r["metric_name"] for r in self.db.get_latest_readings() if r["asset_id"] == asset}
        return next((m for m in LOAD_METRICS if m in have), None)

    def _load_check(self, asset: str, now: float) -> dict[str, Any] | None:
        c = self.cfg
        metric = self._load_metric(asset)
        if metric is None:
            return None
        key = f"load:{asset}"
        ts, load = self._series(asset, metric, c.load_window_days)
        if len(ts) == 0:
            return None
        info = self.meta.get(key)
        if info is not None and any(self._load(f"{key}:{q}") is None for q in ("lo", "mid", "hi")):
            self._forget(key)
            info = None
        due = info is None or now - info["trained_at"] >= c.load_retrain_hours * 3600
        if due and self._span_days(ts) >= c.load_min_days:
            train = ts < ts[-1] - 2 * DAY_S
            X = time_features(ts[train], c.utc_offset_hours)
            started = time.perf_counter()
            for q, alpha in (("lo", 0.1), ("mid", 0.5), ("hi", 0.9)):
                self._store(f"{key}:{q}", fit_quantile(X, load[train], alpha),
                            {"trained_at": now, "rows": int(train.sum()), "kind": "load", "metric": metric})
            self.meta[key] = {"trained_at": now, "rows": int(train.sum()), "kind": "load",
                              "days": round(self._span_days(ts[train]), 1), "metric": metric,
                              "fit_s": round(time.perf_counter() - started, 3)}
            self._save_meta()
            info = self.meta[key]
        if info is None:
            return None
        recent = ts >= now - c.load_alert_slots * SLOT_S - 1
        if recent.sum() < c.load_alert_slots:
            return None
        Xr = time_features(ts[recent], c.utc_offset_hours)
        lo, mid, hi = (predict(self._load(f"{key}:{q}"), Xr) for q in ("lo", "mid", "hi"))
        actual = load[recent]
        # Slots with the output off are an outage (handled by the outage
        # rule), not an unusual load: leave them out.
        out_ts, out_v = self._series(asset, OUTPUT_V, 1)
        if len(out_ts):
            idx = np.clip(np.searchsorted(out_ts, ts[recent]), 0, len(out_ts) - 1)
            powered = out_v[idx] >= 150
            actual, lo, mid, hi = actual[powered], lo[powered], mid[powered], hi[powered]
        high = actual > hi * c.load_high_factor + c.load_high_margin_w
        low = (actual < lo * c.load_low_factor) & (mid > c.load_low_min_expected_w)
        if high.sum() >= c.load_alert_hits:
            return {"code": "load_high", "asset_id": asset, "severity": "warning",
                    "values": {"actual_w": round(float(np.median(actual[high]))),
                               "expected_w": round(float(np.median(mid[high])))}}
        if low.sum() >= c.load_alert_hits:
            return {"code": "load_low", "asset_id": asset, "severity": "warning",
                    "values": {"actual_w": round(float(np.median(actual[low]))),
                               "expected_w": round(float(np.median(mid[low])))}}
        return None

    # ── run-out ──
    def _net_frame(self, asset: str, days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Slots with both state of charge and battery power: (time, SoC %, kW as reported)."""
        soc = self._series(asset, BATTERY_SOC, days)
        p = self._battery_power(asset, days)
        if min(len(soc[0]), len(p[0])) == 0:
            return None
        ts, (ss, pp) = align(soc, p)
        return ts, ss, pp / 1000.0

    def _runout(self, asset: str, now: float) -> dict[str, Any] | None:
        """Warn when the bank is likely to reach its cut-off in the next few hours.

        One model per battery: its usual net discharge (load minus solar) by
        time of day and weekday. No pairing of load, solar and battery assets
        is needed, because the battery's own power already is the difference.
        """
        c = self.cfg
        key = f"runout:{asset}"
        info = self.meta.get(key)
        if info is not None and self._load(key) is None:
            self._forget(key)
            info = None
        due = info is None or now - info["trained_at"] >= c.runout_retrain_hours * 3600
        frame = self._net_frame(asset, c.runout_window_days if due else 7)
        if frame is None or len(frame[0]) < 2:
            return None
        ts, soc, kw = frame
        if due and self._span_days(ts) >= c.runout_min_days:
            step_ok = np.diff(ts) == SLOT_S
            dsoc = np.diff(soc)[step_ok]
            if len(dsoc) < 200 or np.std(dsoc) == 0:
                return None
            # Vendors disagree on the sign of battery power: discharge is the
            # direction that moves state of charge down.
            corr = np.corrcoef(kw[1:][step_ok], dsoc)[0, 1]
            sign = -1.0 if corr > 0 else 1.0
            dis = sign * kw
            # Watt-hours per percent of charge, from every discharging step.
            mean_kw = ((dis[1:] + dis[:-1]) / 2)[step_ok]
            out = mean_kw > 0.05
            swing = float(-dsoc[out].sum())
            if swing < c.runout_min_soc_swing_pct:
                return None
            wh_per_pct = float(mean_kw[out].sum() * 1000 * SLOT_S / 3600) / swing
            # Leave out slots at the cut-off: there the output is off and the
            # battery gives nothing, which is not what a normal night looks like.
            live = soc > c.runout_soc_pct + 2
            if live.sum() < 200 or wh_per_pct <= 0:
                return None
            started = time.perf_counter()
            model = fit_quantile(time_features(ts[live], c.utc_offset_hours), dis[live], 0.5)
            self._store(key, model, {
                "trained_at": now, "rows": int(live.sum()), "days": round(self._span_days(ts), 1),
                "discharge_sign": sign, "wh_per_soc_pct": round(wh_per_pct, 1),
                "usable_kwh_estimate": round(wh_per_pct * 100 / 1000, 1),
                "kind": "runout", "fit_s": round(time.perf_counter() - started, 3)})
            info = self.meta[key]
        model = self._load(key)
        if model is None or info is None:
            return None
        if now - ts[-1] > 3 * SLOT_S:  # no fresh reading: do not guess
            return None
        soc_now = float(soc[-1])
        if soc_now <= c.runout_soc_pct:
            return None  # already there; the low-battery rules speak for it
        week = ts >= now - 7 * DAY_S
        empty_days = np.unique(local_day(ts[week], c.utc_offset_hours)[soc[week] <= c.runout_soc_pct + 2])
        if len(empty_days) >= c.runout_routine_days:
            return None
        n = int(c.runout_horizon_hours * 3600 / SLOT_S)
        future = (np.floor(now / SLOT_S) + 1 + np.arange(n)) * SLOT_S
        pred = predict(model, time_features(future, c.utc_offset_hours))
        level = np.minimum(100.0, soc_now - np.cumsum(pred) * 1000 * SLOT_S / 3600 / info["wh_per_soc_pct"])
        # (the cap at 100% only matters once the sun is back, after any run-out)
        below = np.nonzero(level <= c.runout_soc_pct)[0]
        if len(below) == 0:
            return None
        k = int(below[0])
        trough_end = k + int(np.argmax(pred[k:] <= 0)) if (pred[k:] <= 0).any() else n - 1
        if float(level[k:trough_end + 1].min()) > c.runout_soc_pct - c.runout_margin_pct:
            return None
        hours = (k + 1) * SLOT_S / 3600
        if hours > c.runout_alert_hours:
            return None
        at = time.strftime("%H:%M", time.gmtime(future[k] + c.utc_offset_hours * 3600))
        return {"code": "battery_runout", "asset_id": asset, "severity": "warning",
                "values": {"at": at, "hours": round(hours, 1), "soc": round(soc_now)}}

    # ── entry point ──
    def _each(self, kind: str, assets: list[str], check: Any, now: float,
              findings: list[dict[str, Any]]) -> None:
        for asset in assets:
            try:
                f = check(asset, now)
            except Exception:
                # One odd asset must not silence the findings for the others.
                logger.exception("site_learning_check_failed", kind=kind, asset=asset)
                continue
            if f:
                findings.append(f)

    def step(self, now: float | None = None) -> list[dict[str, Any]]:
        """Train what is due, then return current findings (warnings)."""
        now = time.time() if now is None else now
        findings: list[dict[str, Any]] = []
        self._each("solar", self._pv_assets(), self._pv, now, findings)
        self._each("battery", self._assets_with(BATTERY_V), self._battery, now, findings)
        load_assets: set[str] = set()
        for m in LOAD_METRICS:
            load_assets |= set(self._assets_with(m))
        self._each("load", sorted(load_assets), self._load_check, now, findings)
        self._each("runout", self._assets_with(BATTERY_SOC), self._runout, now, findings)
        return findings
