"""Estimate how much energy Gen318 Edge itself uses, and its share of the site.

    python tools/measure_energy.py --minutes 60 --site-kwh-per-day 40

Run it ON the box while the full stack is running. It samples CPU load and
converts it to watts with a linear model between the board's idle and full
load power. Defaults are for a Raspberry Pi 5 (about 3.0 W idle, 8.8 W with
all cores busy; raspberry.tips, 2026). Add your modem and RS-485 adapters
with --extra-watts. For a measured figure, read a USB power meter instead and
pass it with --measured-watts.
"""

from __future__ import annotations

import argparse
import time

import psutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--idle-watts", type=float, default=3.0)
    ap.add_argument("--full-watts", type=float, default=8.8)
    ap.add_argument("--extra-watts", type=float, default=1.0, help="modem, adapters, SSD")
    ap.add_argument("--measured-watts", type=float, default=None)
    ap.add_argument("--site-kwh-per-day", type=float, default=None)
    a = ap.parse_args()

    if a.measured_watts is not None:
        watts, method = a.measured_watts, "measured with a power meter"
    else:
        samples = []
        end = time.monotonic() + a.minutes * 60
        while time.monotonic() < end:
            samples.append(psutil.cpu_percent(interval=5))
        load = sum(samples) / len(samples) / 100
        watts = a.idle_watts + load * (a.full_watts - a.idle_watts) + a.extra_watts
        method = f"estimated from average CPU load {load:.1%} over {a.minutes:g} min"

    wh_day = watts * 24
    print(f"Average power: {watts:.2f} W ({method})")
    print(f"Energy per day: {wh_day:.0f} Wh ({wh_day * 365 / 1000:.0f} kWh per year)")
    if a.site_kwh_per_day:
        print(f"Share of site energy: {100 * wh_day / (a.site_kwh_per_day * 1000):.2f}%")


if __name__ == "__main__":
    main()
