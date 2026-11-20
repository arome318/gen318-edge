"""Measure on-site learning on simulated sites with injected faults.

    python tools/sitelearn_eval.py --seeds 10

For each scenario and seed it simulates 60 days, steps the learner every
6 hours (hourly around night-time theft) and records how many days after a
fault starts it is first reported, and any finding that is not the injected
fault (a false alarm). Results depend on the simulator: they show the
method works, not how it will perform on a real site.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "edge"))

from inference.sitelearn import SiteLearner  # noqa: E402
from inference.sitesim import DAY_S, T0, Faults, SimHistory, SimSite  # noqa: E402

SCENARIOS = {
    "clean": (Faults(), None, None),
    "panels lose 40%": (Faults(soiling_day=35, soiling_factor=0.6), 35, "pv_underperform"),
    "panels lose 20%": (Faults(soiling_day=35, soiling_factor=0.8), 35, "pv_underperform"),
    "battery resistance x2": (Faults(resistance_day=30, resistance_mult=2.0), 30, "battery_degrading"),
    "battery resistance x1.5": (Faults(resistance_day=30, resistance_mult=1.5), 30, "battery_degrading"),
    "800 W night load": (Faults(theft_day=30, theft_w=800), 30, "load_high"),
    "400 W night load": (Faults(theft_day=30, theft_w=400), 30, "load_high"),
}


def run_one(faults: Faults, onset: float | None, code: str | None, seed: int, night: bool):
    h = SimHistory(SimSite(days=60, seed=seed, faults=faults).run())
    sl = SiteLearner(h, tempfile.mkdtemp())
    times = set(np.arange(T0 + 7 * DAY_S, T0 + 60 * DAY_S, 6 * 3600))
    if night:
        times |= set(np.arange(T0 + 29 * DAY_S, T0 + 34 * DAY_S, 3600))
    delay, false = None, 0
    for t in sorted(times):
        h.now = float(t)
        day = (t - T0) / DAY_S
        for f in sl.step(float(t)):
            if code and f["code"] == code and onset is not None and day >= onset:
                delay = delay if delay is not None else day - onset
            else:
                false += 1
    return delay, false


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    print(f"{'scenario':<26}{'detected':>10}{'median days':>13}{'false alarms':>14}")
    for name, (faults, onset, code) in SCENARIOS.items():
        delays, falses = [], 0
        for seed in range(args.seeds):
            d, f = run_one(faults, onset, code, seed, night="night" in name)
            falses += f
            if d is not None:
                delays.append(d)
        det = "-" if code is None else f"{len(delays)}/{args.seeds}"
        med = f"{np.median(delays):.1f}" if delays else "-"
        print(f"{name:<26}{det:>10}{med:>13}{falses:>14}")


if __name__ == "__main__":
    main()
