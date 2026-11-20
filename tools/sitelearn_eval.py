"""Measure on-site learning on simulated sites with injected faults.

    python tools/sitelearn_eval.py --seeds 10

For each scenario and seed it simulates 60 days, steps the learner every
6 hours (hourly around night-time theft) and records how many days after a
fault starts it is first reported, and any finding that is not the injected
fault (a false alarm). It then checks the two things the simulator can say
about the newer features: that a fall in daytime load (panels throttled by a
full bank) is not reported as dirty panels, and how often and how early the
run-out warning arrives on sites whose bank only sometimes lasts the night.
Results depend on the simulator: they show the method works, not how it
will perform on a real site.
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


def load_drop(seed: int, drop_day: int = 35, factor: float = 0.5) -> int:
    """Daytime customers leave on ``drop_day``; panels are healthy. Count findings."""
    site = SimSite(days=60, seed=seed, load_scale_from=(drop_day, factor))
    h = SimHistory(site.run())
    sl = SiteLearner(h, tempfile.mkdtemp())
    wrong = 0
    for t in np.arange(T0 + 7 * DAY_S, T0 + 60 * DAY_S, 6 * 3600):
        h.now = float(t)
        wrong += sum(f["code"] == "pv_underperform" for f in sl.step(float(t)))
    return wrong


def run_out(seed: int, bank_kwh: float) -> tuple[int, list[float], int, int]:
    """(power cuts from an empty bank, hours of warning for those warned,
    warning episodes, episodes not followed by a bank at or below 15% within 8 hours)."""
    data = SimSite(days=60, seed=seed, bank_kwh=bank_kwh).run()
    ts, soc = data["battery.soc_pct"]
    _, out_v = data["inverter.ac_output_voltage_v"]
    h = SimHistory(data)
    sl = SiteLearner(h, tempfile.mkdtemp())
    sl._pv = sl._battery = sl._load_check = lambda asset, now: None  # run-out only
    warned: list[float] = []
    for t in np.arange(T0 + 9 * DAY_S, T0 + 60 * DAY_S, 1800):
        h.now = float(t)
        if any(f["code"] == "battery_runout" for f in sl.step(float(t))):
            warned.append(float(t))
    off = out_v < 150
    crossings = [c for c in ts[1:][off[1:] & ~off[:-1]] if c > T0 + 9 * DAY_S]
    leads = []
    for c in crossings:
        before = [w for w in warned if c - 8 * 3600 <= w < c]
        if before:
            leads.append((c - min(before)) / 3600)
    episodes, wrong, last = 0, 0, -1e18
    for w in warned:
        if w - last > 6 * 3600:
            episodes += 1
            wrong += int(soc[(ts >= w) & (ts <= w + 8 * 3600)].min() > 15)
        last = w
    return len(crossings), leads, episodes, wrong


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

    wrong = sum(load_drop(seed) for seed in range(args.seeds))
    print(f"\ndaytime load halves, panels healthy: {wrong} dirty-panel findings in {args.seeds} sites")

    print(f"\n{'run-out warning':<26}{'power cuts':>14}{'warned':>8}{'median h':>10}{'warnings':>10}{'wrong':>7}")
    for bank in (17.0, 18.0, 20.0, 24.0):
        n, leads, episodes, wrong = 0, [], 0, 0
        for seed in range(args.seeds):
            a, b, c, d = run_out(seed, bank)
            n, leads, episodes, wrong = n + a, leads + b, episodes + c, wrong + d
        med = f"{np.median(leads):.1f}" if leads else "-"
        print(f"{f'{bank:g} kWh bank':<26}{n:>14}{len(leads):>8}{med:>10}{episodes:>10}{wrong:>7}")


if __name__ == "__main__":
    main()
