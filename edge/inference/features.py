"""Offline feature extraction mirroring cloud PM feature contracts.

The edge intentionally does not import backend code.  These functions derive
the same named, ordered feature contracts from local SQLite history using only
NumPy, so inference remains available with no cloud or pandas dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from common.database import EdgeDatabase

_HOUR_MS = 3_600_000


def _history(
    db: EdgeDatabase, asset_id: str, metric: str, hours: int
) -> list[dict[str, Any]]:
    return [
        row
        for row in db.get_readings_history(asset_id, metric, hours=hours)
        if int(row.get("quality", 4)) <= 2
        and np.isfinite(float(row.get("value", np.nan)))
    ]


def _values(rows: list[dict[str, Any]], hours: int | None = None) -> np.ndarray:
    if not rows:
        return np.array([], dtype=np.float64)
    selected = rows
    if hours is not None:
        cutoff = int(rows[-1]["timestamp"]) - hours * _HOUR_MS
        selected = [row for row in rows if int(row["timestamp"]) >= cutoff]
    return np.asarray([float(row["value"]) for row in selected], dtype=np.float64)


def _mean(values: np.ndarray, default: float = 0.0) -> float:
    return float(np.mean(values)) if len(values) else default


def _std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _trend(values: np.ndarray) -> float:
    if len(values) < 2 or np.std(values) < 1e-12:
        return 0.0
    return float(np.polyfit(np.arange(len(values), dtype=np.float64), values, 1)[0])


def _integrated_ah(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 0.0
    timestamps = np.asarray([int(row["timestamp"]) for row in rows], dtype=np.float64)
    current = np.abs(_values(rows))
    dt_hours = np.diff(timestamps) / 3_600_000.0
    # A long telemetry gap is not evidence that the last current persisted.
    dt_hours = np.clip(dt_hours, 0.0, 0.25)
    return float(np.sum((current[:-1] + current[1:]) * 0.5 * dt_hours))


def _discharge_depths(soc: np.ndarray) -> np.ndarray:
    if len(soc) < 3:
        return np.array([], dtype=np.float64)
    delta = np.diff(soc)
    depths: list[float] = []
    peak = float(soc[0])
    falling = False
    for index, change in enumerate(delta, start=1):
        value = float(soc[index])
        if change < -0.05:
            if not falling:
                peak = float(soc[index - 1])
            falling = True
        elif change > 0.05 and falling:
            depths.append(max(peak - float(soc[index - 1]), 0.0))
            falling = False
        elif not falling:
            peak = max(peak, value)
    if falling:
        depths.append(max(peak - float(soc[-1]), 0.0))
    return np.asarray([depth for depth in depths if depth > 0.5], dtype=np.float64)


def build_battery_health_features(db: EdgeDatabase, asset_id: str) -> dict[str, float]:
    hours = 24 * 14
    current_rows = _history(db, asset_id, "battery.current_a", hours)
    voltage_rows = _history(db, asset_id, "battery.voltage_v", hours)
    temperature_rows = _history(db, asset_id, "battery.temperature_c", hours)
    soc_rows = _history(db, asset_id, "battery.soc_pct", hours)
    cell_min_rows = _history(db, asset_id, "battery.cell_min_voltage_v", hours)
    cell_max_rows = _history(db, asset_id, "battery.cell_max_voltage_v", hours)

    current = _values(current_rows)
    voltage = _values(voltage_rows)
    temperature = _values(temperature_rows)
    soc = _values(soc_rows)
    depths = _discharge_depths(soc)
    throughput = _integrated_ah(current_rows)
    capacity_rows = _history(db, asset_id, "battery.capacity_ah", hours)
    capacity_ah = _mean(_values(capacity_rows), 100.0)

    if temperature_rows:
        timestamps = np.asarray(
            [int(row["timestamp"]) for row in temperature_rows], dtype=np.float64
        )
        dt_hours = np.clip(np.diff(timestamps) / 3_600_000.0, 0.0, 0.25)
        thermal_stress = float(
            np.sum(np.maximum(temperature[:-1] - 35.0, 0.0) * dt_hours)
        )
    else:
        thermal_stress = 0.0

    spread_count = min(len(cell_min_rows), len(cell_max_rows))
    spreads = (
        _values(cell_max_rows)[-spread_count:] - _values(cell_min_rows)[-spread_count:]
        if spread_count
        else np.array([], dtype=np.float64)
    )
    charge = current[current > 0.1]
    relaxation_count = min(len(current), len(voltage))
    relaxation_voltage = np.array([], dtype=np.float64)
    if relaxation_count:
        aligned_current = current[-relaxation_count:]
        aligned_voltage = voltage[-relaxation_count:]
        relaxation_voltage = aligned_voltage[np.abs(aligned_current) < 0.1]

    # ICA/DVA proxies use aligned local samples. They are deliberately robust
    # and return zero when the required charge/discharge segment is absent.
    pair_count = min(len(current), len(voltage))
    ica = np.array([], dtype=np.float64)
    dva = np.array([], dtype=np.float64)
    if pair_count > 2:
        aligned_current = current[-pair_count:]
        aligned_voltage = voltage[-pair_count:]
        dv = np.diff(aligned_voltage)
        safe_dv = np.where(np.abs(dv) < 1e-6, np.nan, dv)
        ica = aligned_current[1:] / safe_dv
        discharge = aligned_current[1:] < -0.1
        dva = dv[discharge] / np.maximum(np.abs(aligned_current[1:][discharge]), 1e-6)
        ica = ica[np.isfinite(ica) & (aligned_current[1:] > 0.1)]
        dva = dva[np.isfinite(dva)]

    timestamps = [
        int(row["timestamp"]) for row in current_rows + voltage_rows + soc_rows
    ]
    age_days = (max(timestamps) - min(timestamps)) / 86_400_000.0 if timestamps else 0.0
    # Voltage droop under load. MUST mirror the cloud
    # (backend/app/ml/features/battery_features.py section 7), which trains
    # the model this feature is served to: split on the 90th percentile of
    # |current| and difference the two voltage means.
    #
    # This previously used an unconditioned percentile spread of all voltage,
    # which reports the STATE-OF-CHARGE swing rather than load droop — on a
    # bank cycling 48-56 V that is ~8 V against a true droop under 2 V. Same
    # feature name, different physical quantity, no test comparing the two.
    voltage_sag = 0.0
    sag_count = min(len(current), len(voltage))
    if sag_count:
        aligned_current = np.abs(current[-sag_count:])
        aligned_voltage = voltage[-sag_count:]
        high_q = float(np.quantile(aligned_current, 0.9))
        under_load = aligned_current > high_q
        if under_load.any() and (~under_load).any():
            voltage_sag = float(
                np.mean(aligned_voltage[~under_load])
                - np.mean(aligned_voltage[under_load])
            )
    return {
        "cumulative_ah_throughput": throughput,
        "estimated_cycles": throughput / max(2.0 * capacity_ah, 1.0),
        "temp_stress_degree_hours": thermal_stress,
        "avg_temperature_c": _mean(temperature, 30.0),
        "max_temperature_c": float(np.max(temperature)) if len(temperature) else 30.0,
        "avg_discharge_depth_pct": _mean(depths),
        "max_discharge_depth_pct": float(np.max(depths)) if len(depths) else 0.0,
        "deep_discharge_count": float(np.sum(depths >= 80.0)),
        "cell_voltage_spread_avg_v": _mean(spreads),
        "cell_voltage_spread_max_v": float(np.max(spreads)) if len(spreads) else 0.0,
        "voltage_sag_v": voltage_sag,
        "chemistry_calendar_factor": 1.0,
        "chemistry_dod_sensitivity": 0.3,
        "chemistry_cycle_life_nominal": 4000.0,
        "age_days": age_days,
        "avg_charge_rate_a": _mean(charge),
        "max_charge_rate_a": float(np.max(charge)) if len(charge) else 0.0,
        "ica_peak_height": float(np.max(np.abs(ica))) if len(ica) else 0.0,
        "ica_peak_voltage": float(np.median(voltage))
        if len(ica) and len(voltage)
        else 0.0,
        "ica_area": float(np.sum(np.abs(ica))) if len(ica) else 0.0,
        "dva_plateau_length": float(np.sum(np.abs(dva) < 0.01)) if len(dva) else 0.0,
        "dva_slope_mean": _mean(dva),
        "relaxation_voltage_std": _std(relaxation_voltage),
        "relaxation_voltage_drift": _trend(relaxation_voltage),
        # Cloud formula (battery_features.py section 14):
        # relaxation_voltage_std / sqrt(age_days), zero when either is zero.
        # The edge previously used voltage_sag * thermal_stress / age_days —
        # a different formula under the same feature name, feeding a model
        # trained on the cloud one.
        "sei_growth_proxy": (
            _std(relaxation_voltage) / (age_days**0.5)
            if age_days > 0 and _std(relaxation_voltage) > 0
            else 0.0
        ),
    }


def build_battery_rul_features(
    db: EdgeDatabase, asset_id: str, health: dict[str, float]
) -> dict[str, float]:
    soh_rows = _history(db, asset_id, "battery.soh_pct", 24 * 30)
    soh = _values(soh_rows)
    current_soh = float(soh[-1]) if len(soh) else 100.0
    decline = max(float(soh[0] - soh[-1]), 0.0) / 30.0 if len(soh) > 1 else 0.0
    avg_temp = health["avg_temperature_c"]
    # Arrhenius-shaped, normalised calendar acceleration proxy.
    calendar_factor = float(np.exp(np.clip((avg_temp - 25.0) / 20.0, -2.0, 3.0)))
    return {
        "calendar_aging_factor": calendar_factor,
        "sei_growth_rate": health["sei_growth_proxy"],
        "cycle_count_weighted": health["estimated_cycles"],
        "avg_temperature_c": avg_temp,
        "max_temperature_c": health["max_temperature_c"],
        "avg_depth_of_discharge_pct": health["avg_discharge_depth_pct"],
        "deep_discharge_count": health["deep_discharge_count"],
        "current_soh_pct": current_soh,
        "soh_decline_rate_30d": decline,
        "cell_voltage_spread_avg_v": health["cell_voltage_spread_avg_v"],
        "voltage_sag_v": health["voltage_sag_v"],
        "age_days": health["age_days"],
        "chemistry_cycle_life_nominal": health["chemistry_cycle_life_nominal"],
    }


def build_inverter_features(
    db: EdgeDatabase, asset_id: str, rated_power_w: float = 0.0
) -> dict[str, float]:
    def metric(name: str) -> list[dict[str, Any]]:
        return _history(db, asset_id, name, 24 * 7)

    efficiency = metric("inverter.efficiency_pct")
    temperature = metric("inverter.temperature_c")
    voltage = metric("inverter.ac_output_voltage_v")
    current = metric("inverter.ac_output_current_a")
    frequency = metric("inverter.ac_output_freq_hz")
    errors = metric("inverter.error_code")
    power_factor = metric("inverter.power_factor")
    power = metric("inverter.ac_output_power_w")
    timeouts = metric("inverter.comm_timeout")
    gaps = metric("inverter.comm_gap_seconds")

    eff_1h, eff_24h, eff_7d = (
        _values(efficiency, 1),
        _values(efficiency, 24),
        _values(efficiency),
    )
    temp_1h, temp_24h, temp_7d = (
        _values(temperature, 1),
        _values(temperature, 24),
        _values(temperature),
    )
    voltage_1h, voltage_24h = _values(voltage, 1), _values(voltage, 24)
    current_1h = _values(current, 1)
    frequency_1h, frequency_24h = _values(frequency, 1), _values(frequency, 24)
    pf_1h = _values(power_factor, 1)
    power_1h, power_24h = _values(power, 1), _values(power, 24)
    correlation_count = min(len(temp_24h), len(power_24h))
    correlation = 0.0
    if correlation_count > 10:
        candidate = np.corrcoef(
            temp_24h[-correlation_count:], power_24h[-correlation_count:]
        )[0, 1]
        correlation = float(candidate) if np.isfinite(candidate) else 0.0

    return {
        "efficiency_1h_mean": _mean(eff_1h),
        "efficiency_1h_std": _std(eff_1h),
        "efficiency_24h_mean": _mean(eff_24h),
        "efficiency_24h_std": _std(eff_24h),
        "efficiency_7d_mean": _mean(eff_7d),
        "efficiency_7d_trend": _trend(eff_7d),
        "temperature_1h_mean": _mean(temp_1h),
        "temperature_1h_max": float(np.max(temp_1h)) if len(temp_1h) else 0.0,
        "temperature_24h_mean": _mean(temp_24h),
        "temperature_7d_trend": _trend(temp_7d),
        "temp_vs_load_correlation": correlation,
        "ac_output_voltage_1h_std": _std(voltage_1h),
        "ac_output_voltage_24h_range": (
            float(np.ptp(voltage_24h)) if len(voltage_24h) else 0.0
        ),
        "voltage_thd_estimate": _std(voltage_1h) / max(_mean(voltage_1h), 1.0),
        "ac_output_current_1h_std": _std(current_1h),
        "current_thd_estimate": _std(current_1h) / max(_mean(current_1h), 1.0),
        "frequency_1h_std": _std(frequency_1h),
        "frequency_24h_range": (
            float(np.ptp(frequency_24h)) if len(frequency_24h) else 0.0
        ),
        "error_code_frequency_24h": float(np.sum(_values(errors, 24) > 0)),
        "unique_error_codes_7d": float(len(np.unique(_values(errors)))),
        "power_factor_1h_mean": _mean(pf_1h),
        "power_factor_1h_std": _std(pf_1h),
        "load_ratio_1h_mean": _mean(power_1h) / rated_power_w
        if rated_power_w > 0
        else 0.0,
        "overload_minutes_24h": (
            float(np.sum(power_24h > rated_power_w)) if rated_power_w > 0 else 0.0
        ),
        "comm_timeout_count_24h": float(np.sum(_values(timeouts, 24))),
        "comm_gap_max_seconds_24h": (
            float(np.max(_values(gaps, 24))) if len(_values(gaps, 24)) else 0.0
        ),
    }
