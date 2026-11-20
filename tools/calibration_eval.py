"""Could the box set its own alarm thresholds to a stated false-alarm rate?

    python tools/calibration_eval.py --seeds 10

An experiment, not part of the box: nothing here changes when alerts fire.

Today the unusual-load warning uses hand-set numbers (30% and 300 W above
the learned band, in 6 of 8 quarter-hours). This asks what happens if the
box instead *measures* how far its own site strays from the band on normal
days, and puts the threshold where only a chosen share of normal windows
would cross it. That is split conformal calibration:

1. learn the load band from the first 14 days (as the box does now);
2. on the next 14 days, score every 2-hour window by its 6th-largest
   excess over the band (so "6 of 8 quarter-hours above X" becomes
   "the window's score is above X");
3. set the threshold at the level that only a share ``alpha`` of those
   normal windows exceeded.

Then it counts, on days 30 to 60 of simulated sites, how often that
threshold fires on a clean site and whether it catches an extra night load,
next to today's fixed rule.

Read the result with care. The simulator's load noise is the same kind
every day, which is exactly the condition under which this method's
promise holds. A real village has market days, festivals and seasons; the
promise weakens when tomorrow is not like the calibration days. This shows
the idea is worth a field trial, not that it works in the field.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "edge"))

from inference.sitelearn import SLOT_S, fit_quantile, predict, time_features  # noqa: E402
from inference.sitesim import DAY_S, T0, Faults, SimSite  # noqa: E402

WINDOW, HITS = 8, 6          # the box's rule: 6 of 8 quarter-hours
TRAIN_DAYS, CAL_DAYS, TEST_FROM = 14, 14, 30


def slots(ts: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.floor(ts / SLOT_S) * SLOT_S
    uniq, inv = np.unique(s, return_inverse=True)
    return uniq, np.bincount(inv, weights=v) / np.bincount(inv)


def window_scores(excess: np.ndarray) -> np.ndarray:
    """For each slot, the HITS-th largest excess among the WINDOW slots ending there."""
    out = np.full(len(excess), -np.inf)
    for i in range(WINDOW - 1, len(excess)):
        out[i] = np.sort(excess[i - WINDOW + 1:i + 1])[-HITS]
    return out


def episodes(fired: np.ndarray) -> list[int]:
    """Start index of each run of alarms (runs closer than one window count as one)."""
    starts, last = [], -10**9
    for i in np.nonzero(fired)[0]:
        if i - last > WINDOW:
            starts.append(int(i))
        last = i
    return starts


def one_site(seed: int, theft_w: float | None, alphas: list[float]) -> dict:
    faults = Faults(theft_day=TEST_FROM, theft_w=theft_w) if theft_w else Faults()
    data = SimSite(days=60, seed=seed, faults=faults).run()
    ts, load = slots(*data["system.ac_consumption_w"])
    _, out_v = slots(*data["inverter.ac_output_voltage_v"])
    day = (ts - T0) / DAY_S
    X = time_features(ts, 1.0)
    train = day < TRAIN_DAYS
    hi = predict(fit_quantile(X[train], load[train], 0.9), X)
    powered = out_v >= 150
    excess = np.where(powered, load - hi, -np.inf)       # watts above the learned band
    score = window_scores(excess)
    cal = (day >= TRAIN_DAYS) & (day < TRAIN_DAYS + CAL_DAYS) & np.isfinite(score)
    test = (day >= TEST_FROM) & np.isfinite(score)
    cal_scores = np.sort(score[cal])
    n = len(cal_scores)
    out = {}
    for alpha in alphas:
        k = math.ceil((n + 1) * (1 - alpha))             # conformal rank; needs k <= n
        tau = cal_scores[min(k, n) - 1]
        out[f"alpha {alpha:g}"] = (score > tau) & test
    # today's rule, in the same frame: 6 of 8 slots above 1.3 x band + 300 W
    fixed_excess = np.where(powered, load - (1.3 * hi + 300.0), -np.inf)
    out["fixed rule"] = (window_scores(fixed_excess) > 0) & test
    result = {}
    for name, fired in out.items():
        starts = episodes(fired)
        first = (ts[starts[0]] - (T0 + TEST_FROM * DAY_S)) / 3600 if starts else None
        result[name] = {"windows": int(test.sum()), "fired": int(fired.sum()), "episodes": len(starts),
                        "first_hours": first}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    alphas = [0.01, 0.001]
    names = [f"alpha {x:g}" for x in alphas] + ["fixed rule"]
    runs = {label: [one_site(s, w, alphas) for s in range(a.seeds)]
            for label, w in (("clean", None), ("400 W night load", 400.0), ("800 W night load", 800.0))}
    days = 60 - TEST_FROM
    print(f"{a.seeds} simulated sites, tested on days {TEST_FROM} to 60 ({a.seeds * days} site-days per row)\n")
    print(f"{'threshold':<14}{'windows over it, clean':>24}{'false alarms / 30 days':>24}"
          f"{'400 W caught':>14}{'hours':>7}{'800 W caught':>14}{'hours':>7}")
    for name in names:
        clean = [r[name] for r in runs["clean"]]
        rate = 100 * sum(c["fired"] for c in clean) / sum(c["windows"] for c in clean)
        per30 = sum(c["episodes"] for c in clean) / a.seeds * 30 / days
        cells = []
        for label in ("400 W night load", "800 W night load"):
            # caught = an alarm in the first 24 hours of the extra load
            firsts = [r[name]["first_hours"] for r in runs[label]]
            hit = [f for f in firsts if f is not None and f <= 24]
            cells += [f"{len(hit)}/{a.seeds}", f"{np.median(hit):.1f}" if hit else "-"]
        print(f"{name:<14}{rate:>23.2f}%{per30:>24.1f}{cells[0]:>14}{cells[1]:>7}{cells[2]:>14}{cells[3]:>7}")
    print("\n'alpha' is the share of normal windows the threshold is meant to let through "
          "(0.01 = 1%).\nSimulator only: see the note at the top of this file.")


if __name__ == "__main__":
    main()
