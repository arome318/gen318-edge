"""Turn a box's alert record into rows for an open fault dataset.

Each row is one alert the box raised, what the technician said about it
(real, false alarm, other cause, or nothing), and what the site looked like
in the hour before. The rows are made to be shared, so they leave out
anything that points at a person or a place:

* no phone numbers (the box never stores them), and no technician notes,
  which are free text and can contain names;
* no site name, device id or asset id: the site is a made-up id that is the
  same for every export from one box and cannot be turned back into it;
* no exact times: the month, the weekday and the hour of day are kept,
  which is what a model needs, and the date is not.

See docs/FAULT_DATASET.md for what the dataset is and is not good for.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

SCHEMA = "gen318-edge/fault-dataset/1"
LABELS = {1: "real", 2: "false_alarm", 3: "other_cause"}
CONTEXT_S = 3600
# The hour before the alert, averaged. Solar and load are added up across
# assets by the caller's readings; the rest are averaged.
CONTEXT = {
    "battery_soc_pct": ("battery.soc_pct",),
    "battery_voltage_v": ("battery.voltage_v",),
    "battery_power_w": ("battery.power_w",),
    "battery_temperature_c": ("battery.temperature_c",),
    "solar_power_w": ("solar.pv_power_w", "solar.mppt1_power_w", "solar.mppt2_power_w",
                      "solar.mppt3_power_w", "solar.mppt4_power_w"),
    "load_power_w": ("system.ac_consumption_w", "load.power_w", "inverter.ac_output_power_w"),
    "inverter_temperature_c": ("inverter.temperature_c",),
    "output_voltage_v": ("inverter.ac_output_voltage_v",),
    "genset_power_w": ("genset.power_w",),
}
COLUMNS = ("site", "code", "severity", "month", "weekday", "hour", "minutes_open", "label",
           "minutes_to_label", *(f"value_{k}" for k in ("soc", "temp", "pct", "drop_pct", "actual_w", "expected_w")),
           *(f"before_{k}" for k in CONTEXT))


def _context(db: Any | None, raised_at: float) -> dict[str, float | None]:
    out: dict[str, float | None] = {k: None for k in CONTEXT}
    if db is None:
        return out
    metrics = tuple(m for names in CONTEXT.values() for m in names)
    series = db.series_between(int((raised_at - CONTEXT_S) * 1000), int(raised_at * 1000), metrics)
    for key, names in CONTEXT.items():
        # first metric name that has data wins (a total, else its parts)
        for i, name in enumerate(names):
            found = [vals for (_, metric), (_, vals) in series.items() if metric == name and len(vals)]
            if key == "solar_power_w" and i > 0:
                found = [vals for (_, metric), (_, vals) in series.items() if metric in names[1:] and len(vals)]
            if found:
                means = [float(np.mean(v)) for v in found]
                total = key in ("solar_power_w", "load_power_w", "genset_power_w", "battery_power_w")
                out[key] = round(sum(means) if total else float(np.mean(means)), 2)
                break
    return out


def build_rows(store: Any, db: Any | None, device_id: str, utc_offset_hours: float = 1.0,
               labelled_only: bool = False) -> list[dict[str, Any]]:
    site = store.site_pseudonym(device_id)
    rows = []
    for a in store.fault_rows():
        if labelled_only and a["label"] is None:
            continue
        local = time.gmtime(a["raised_at"] + utc_offset_hours * 3600)
        rows.append({
            "site": site,
            "code": a["code"],
            "severity": a["severity"],
            "month": time.strftime("%Y-%m", local),
            "weekday": local.tm_wday,                 # Monday = 0
            "hour": local.tm_hour,
            "minutes_open": round((a["cleared_at"] - a["raised_at"]) / 60, 1) if a["cleared_at"] else None,
            "label": LABELS.get(a["label"]),
            "minutes_to_label": (round((a["labelled_at"] - a["raised_at"]) / 60, 1)
                                 if a["labelled_at"] is not None else None),
            "values": {k: v for k, v in (a["values"] or {}).items() if isinstance(v, (int, float))},
            "before": _context(db, a["raised_at"]),
        })
    return rows


def to_csv(rows: list[dict[str, Any]]) -> str:
    def cell(value: Any) -> str:
        return "" if value is None else str(value)

    lines = [",".join(COLUMNS)]
    for r in rows:
        flat = {**{k: r[k] for k in COLUMNS[:9]},
                **{f"value_{k}": v for k, v in r["values"].items()},
                **{f"before_{k}": v for k, v in r["before"].items()}}
        lines.append(",".join(cell(flat.get(c)) for c in COLUMNS))
    return "\n".join(lines) + "\n"


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [r["label"] for r in rows if r["label"]]
    by_code: dict[str, dict[str, int]] = {}
    for r in rows:
        d = by_code.setdefault(r["code"], {"alerts": 0, "real": 0, "false_alarm": 0, "other_cause": 0})
        d["alerts"] += 1
        if r["label"]:
            d[r["label"]] += 1
    return {"schema": SCHEMA, "alerts": len(rows), "labelled": len(labels),
            "sites": len({r["site"] for r in rows}),
            "months": sorted({r["month"] for r in rows}),
            "labels": {k: labels.count(k) for k in LABELS.values()}, "by_code": by_code}
