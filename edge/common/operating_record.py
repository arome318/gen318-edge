"""A month's operating record for one site, in a plain open format.

Lenders, guarantors and results-based programmes ask operators what a site
actually did: how much of the time power was available, how much energy was
delivered, what went wrong and how it was handled. This builds that record
on the box from its own readings and SMS records. Nothing is sent anywhere;
the operator decides who gets the file.

What the figures mean, and what they do not:

* Time is counted in 15-minute slots. A slot with no reading is *unknown*,
  never counted as power on or off. ``data_coverage_pct`` says how much of
  the month the box actually saw, and every other figure covers only that.
* ``power_available_pct`` is the share of observed slots in which the
  inverter's output voltage was at or above the outage threshold.
* Energy is the average power in each observed slot times a quarter of an
  hour. It is measured at the inverter, not at customer meters, and is only
  as good as the register map it was read through (see
  ``register_map_verification``).
* ``record_sha256`` lets a reader check the file was not altered after it
  was made. It is a checksum, not a signature: it does not prove who made it.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import time
from typing import Any

import numpy as np

SCHEMA = "gen318-edge/operating-record/1"
SLOT_S = 900
SLOT_H = SLOT_S / 3600

OUTPUT_V = "inverter.ac_output_voltage_v"
SOC = "battery.soc_pct"
SOLAR_TOTAL = "solar.pv_power_w"
SOLAR_PARTS = tuple(f"solar.mppt{i}_power_w" for i in range(1, 5))
LOAD = ("system.ac_consumption_w", "load.power_w", "inverter.ac_output_power_w")
GENSET = "genset.power_w"
METRICS = (OUTPUT_V, SOC, SOLAR_TOTAL, *SOLAR_PARTS, *LOAD, GENSET)
LABELS = {1: "real", 2: "false_alarm", 3: "other_cause"}

DAILY_COLUMNS = ("date", "data_coverage_pct", "power_available_pct", "outage_minutes", "energy_delivered_kwh",
                 "solar_kwh", "genset_kwh", "lowest_battery_pct", "alerts_critical", "alerts_warning")


def month_bounds(month: str, utc_offset_hours: float) -> tuple[float, float, int]:
    """(start, end, days) of a ``YYYY-MM`` month in the site's local time, as UTC seconds."""
    year, mon = (int(x) for x in month.split("-"))
    if not (2000 <= year <= 2100 and 1 <= mon <= 12):
        raise ValueError(f"not a month: {month!r} (use YYYY-MM)")
    days = calendar.monthrange(year, mon)[1]
    start = calendar.timegm((year, mon, 1, 0, 0, 0)) - utc_offset_hours * 3600
    return start, start + days * 86400, days


def _slots(ts: np.ndarray, values: np.ndarray, start: float, n: int) -> np.ndarray:
    """Mean value per 15-minute slot from ``start``; NaN where nothing was read."""
    out = np.full(n, np.nan)
    if len(ts) == 0:
        return out
    idx = np.floor((ts - start) / SLOT_S).astype(int)
    keep = (idx >= 0) & (idx < n)
    sums = np.bincount(idx[keep], weights=values[keep], minlength=n)
    counts = np.bincount(idx[keep], minlength=n)
    seen = counts > 0
    out[seen] = sums[seen] / counts[seen]
    return out


def _sum_assets(series: dict[tuple[str, str], tuple[Any, Any]], choose: Any, start: float, n: int) -> np.ndarray:
    """Add up one power figure per asset; NaN only where no asset reported."""
    assets = sorted({a for a, _ in series})
    total, seen = np.zeros(n), np.zeros(n, bool)
    for asset in assets:
        slot = choose(asset)
        if slot is None:
            continue
        ok = np.isfinite(slot)
        total[ok] += slot[ok]
        seen |= ok
    total[~seen] = np.nan
    return total


def _kwh(power_w: np.ndarray) -> float:
    return float(np.nansum(np.clip(power_w, 0, None)) * SLOT_H / 1000)


def _round(value: float | None, digits: int = 1) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def build_record(db: Any, sms_store: Any | None, month: str, *, site_name: str, device_id: str,
                 utc_offset_hours: float = 1.0, outage_voltage_v: float = 150.0,
                 register_map_verification: str = "unknown", now: float | None = None) -> dict[str, Any]:
    """The operating record for ``month`` (``YYYY-MM``, site local time)."""
    now = time.time() if now is None else now
    start, end, days = month_bounds(month, utc_offset_hours)
    n = days * 96
    series = db.series_between(int(start * 1000), int(end * 1000), METRICS)

    def one(asset: str, metric: str) -> np.ndarray | None:
        pair = series.get((asset, metric))
        return _slots(pair[0], pair[1], start, n) if pair is not None else None

    def solar(asset: str) -> np.ndarray | None:
        total = one(asset, SOLAR_TOTAL)
        if total is not None:
            return total
        parts = [p for p in (one(asset, m) for m in SOLAR_PARTS) if p is not None]
        if not parts:
            return None
        stack = np.vstack(parts)
        seen = np.isfinite(stack).any(axis=0)
        return np.where(seen, np.nansum(stack, axis=0), np.nan)   # no tracker read = unknown, not zero

    def load(asset: str) -> np.ndarray | None:
        return next((s for s in (one(asset, m) for m in LOAD) if s is not None), None)

    load_w = _sum_assets(series, load, start, n)
    solar_w = _sum_assets(series, solar, start, n)
    genset_w = _sum_assets(series, lambda a: one(a, GENSET), start, n)
    def across(metric: str, pick: Any) -> np.ndarray:
        """One figure per slot across assets (highest or lowest); NaN where none reported."""
        rows = [s for s in (one(a, metric) for a in sorted({a for a, _ in series})) if s is not None]
        if not rows:
            return np.full(n, np.nan)
        stack = np.vstack(rows)
        seen = np.isfinite(stack).any(axis=0)
        fill = -np.inf if pick is np.max else np.inf
        return np.where(seen, pick(np.where(np.isfinite(stack), stack, fill), axis=0), np.nan)

    out_v = across(OUTPUT_V, np.max)   # the site has power if any inverter is on
    soc = across(SOC, np.min)

    everything = [load_w, solar_w, genset_w, out_v, soc]
    observed = np.any(np.isfinite(np.vstack(everything)), axis=0)
    elapsed = int(np.clip((min(now, end) - start) // SLOT_S, 0, n))   # slots of the month that have happened

    alerts: list[dict[str, Any]] = []
    sms_counts: dict[str, int] = {}
    if sms_store is not None:
        labelled = {a["ref"]: a.get("label") for a in sms_store.recent_alerts(100000)}
        for a in sms_store.alerts_between(start, end):
            cleared = a["cleared_at"]
            alerts.append({
                "ref": a["ref"], "code": a["code"], "severity": a["severity"],
                "raised_at": _iso(a["raised_at"]), "cleared_at": _iso(cleared) if cleared else None,
                "minutes_open": _round((cleared - a["raised_at"]) / 60) if cleared else None,
                "technician_label": LABELS.get(labelled.get(a["ref"])),
            })
        sms_counts = sms_store.delivery_counts(start, end)

    daily = []
    for d in range(days):
        sl = slice(d * 96, (d + 1) * 96)
        day_start = start + d * 86400
        seen = observed[sl]
        v = out_v[sl]
        known = np.isfinite(v)
        on = known & (v >= outage_voltage_v)
        day_alerts = [a for a in alerts if day_start <= _unix(a["raised_at"]) < day_start + 86400]
        daily.append({
            "date": time.strftime("%Y-%m-%d", time.gmtime(day_start + utc_offset_hours * 3600)),
            "data_coverage_pct": _round(100 * seen.mean()),
            "power_available_pct": _round(100 * on.sum() / known.sum()) if known.any() else None,
            "outage_minutes": int((known & ~on).sum() * 15) if known.any() else None,
            "energy_delivered_kwh": _round(_kwh(load_w[sl]), 2) if np.isfinite(load_w[sl]).any() else None,
            "solar_kwh": _round(_kwh(solar_w[sl]), 2) if np.isfinite(solar_w[sl]).any() else None,
            "genset_kwh": _round(_kwh(genset_w[sl]), 2) if np.isfinite(genset_w[sl]).any() else None,
            "lowest_battery_pct": _round(np.nanmin(soc[sl])) if np.isfinite(soc[sl]).any() else None,
            "alerts_critical": sum(a["severity"] == "critical" for a in day_alerts),
            "alerts_warning": sum(a["severity"] == "warning" for a in day_alerts),
        })

    known = np.isfinite(out_v)
    on = known & (out_v >= outage_voltage_v)
    labels = [a["technician_label"] for a in alerts if a["technician_label"]]
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "site_name": site_name,
        "device_id": device_id,
        "month": month,
        "utc_offset_hours": utc_offset_hours,
        "generated_at": _iso(now),
        "month_complete": now >= end,
        "register_map_verification": register_map_verification,
        "outage_voltage_v": outage_voltage_v,
        "totals": {
            "data_coverage_pct": _round(100 * observed[:elapsed].mean()) if elapsed else 0.0,
            "days_with_data": sum(1 for d in daily if d["data_coverage_pct"]),
            "power_available_pct": _round(100 * on.sum() / known.sum(), 2) if known.any() else None,
            "outage_hours": _round((known & ~on).sum() * SLOT_H, 2) if known.any() else None,
            "energy_delivered_kwh": _round(_kwh(load_w), 1) if np.isfinite(load_w).any() else None,
            "solar_kwh": _round(_kwh(solar_w), 1) if np.isfinite(solar_w).any() else None,
            "genset_kwh": _round(_kwh(genset_w), 1) if np.isfinite(genset_w).any() else None,
            "lowest_battery_pct": _round(np.nanmin(soc)) if np.isfinite(soc).any() else None,
            "alerts": {"critical": sum(a["severity"] == "critical" for a in alerts),
                       "warning": sum(a["severity"] == "warning" for a in alerts)},
            "alerts_labelled": {k: labels.count(k) for k in LABELS.values()},
            "sms_sent": sms_counts,
        },
        "daily": daily,
        "alerts": alerts,
        "notes": [
            "Time is counted in 15-minute slots; slots with no reading are unknown, not on or off.",
            "Every figure covers only the share of the month given by data_coverage_pct.",
            "Energy is measured at the inverter, not at customer meters.",
            "record_sha256 is a checksum of this record, not a signature.",
        ],
    }
    record["record_sha256"] = checksum(record)
    return record


def checksum(record: dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify(record: dict[str, Any]) -> bool:
    """True if the record is unchanged since it was made."""
    return record.get("record_sha256") == checksum(record)


def daily_csv(record: dict[str, Any]) -> str:
    lines = [",".join(DAILY_COLUMNS)]
    for day in record["daily"]:
        lines.append(",".join("" if day[c] is None else str(day[c]) for c in DAILY_COLUMNS))
    return "\n".join(lines) + "\n"


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _unix(iso: str) -> float:
    return calendar.timegm(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))
