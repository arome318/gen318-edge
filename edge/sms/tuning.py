"""Let technician replies set how sensitive each warning is at this site.

A technician answers an alert with 1 (real), 2 (false alarm) or 3 (other
cause). Those answers decide a small number of "steps" per warning:

* three or more false-alarm replies among the last five, and no "real"
  reply among them: one step less sensitive;
* five false-alarm replies in a row: two steps (the most allowed);
* any "real" reply among the last five: back to the installer's setting.

The rule looks only at the stored replies, so it needs no state of its own
and undoes itself. Critical alerts (power off, battery critical, battery
hot, inverter fault, lost readings) are never loosened: a technician who is
tired of an alert must not be able to switch off a safety warning by SMS.
"RESET TUNING" from a technician discards the replies counted so far.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from sms.rules import Thresholds

MAX_STEPS = 2
WINDOW = 5

# Warnings decided by the fixed rules in sms/rules.py.
RULE_CODES = ("battery_low", "cell_imbalance", "inverter_hot", "overload", "diesel_waste")
# Warnings decided by the models in inference/sitelearn.py, which applies its
# own share of each step (see sitelearn.tuned_config).
LEARNED_CODES = ("pv_underperform", "battery_degrading", "load_high", "load_low", "battery_runout")
TUNABLE = frozenset(RULE_CODES + LEARNED_CODES)


def steps_from_labels(labels: list[int]) -> int:
    """Steps for one warning from its labels, newest first (1 real, 2 false)."""
    recent = [x for x in labels if x in (1, 2)][:WINDOW]
    if 1 in recent or recent.count(2) < 3:
        return 0
    return MAX_STEPS if recent.count(2) >= WINDOW else 1


def decide(labels_by_code: dict[str, list[int]]) -> dict[str, int]:
    """Steps per warning code; codes at the installer's setting are left out."""
    out = {}
    for code, labels in labels_by_code.items():
        if code in TUNABLE:
            steps = steps_from_labels(labels)
            if steps:
                out[code] = steps
    return out


def tuned_thresholds(base: Thresholds, steps: dict[str, int]) -> Thresholds:
    """The installer's thresholds with each rule warning loosened by its steps."""
    def n(code: str) -> int:
        return max(0, min(MAX_STEPS, int(steps.get(code, 0))))

    return replace(
        base,
        # never closer than 2 points to the critical level
        soc_warning_pct=max(base.soc_warning_pct - 2.5 * n("battery_low"),
                            min(base.soc_warning_pct, base.soc_critical_pct + 2)),
        cell_spread_v=base.cell_spread_v * (1 + 0.2 * n("cell_imbalance")),
        inverter_temp_c=base.inverter_temp_c + 3 * n("inverter_hot"),
        overload_ratio=max(base.overload_ratio, min(1.0, base.overload_ratio + 0.03 * n("overload"))),
        genset_soc_pct=max(base.genset_soc_pct, min(90.0, base.genset_soc_pct + 5 * n("diesel_waste"))),
    )


def describe(base: Thresholds, steps: dict[str, int]) -> list[dict[str, Any]]:
    """Rows for the local page: what changed and by how much."""
    tuned = tuned_thresholds(base, steps)
    fields = {"battery_low": ("soc_warning_pct", "%"), "cell_imbalance": ("cell_spread_v", " V"),
              "inverter_hot": ("inverter_temp_c", " C"), "overload": ("overload_ratio", ""),
              "diesel_waste": ("genset_soc_pct", "%")}
    rows = []
    for code, n in sorted(steps.items()):
        row: dict[str, Any] = {"code": code, "steps": n, "max_steps": MAX_STEPS}
        if code in fields:
            name, unit = fields[code]
            row["was"] = f"{round(getattr(base, name), 3):g}{unit}"
            row["now"] = f"{round(getattr(tuned, name), 3):g}{unit}"
        rows.append(row)
    return rows
