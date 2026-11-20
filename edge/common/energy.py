"""What the box itself costs the site in energy.

A monitoring box on a mini-grid is one more load on a small battery, so it
should say what it uses. The watchdog samples CPU load once a minute and
keeps a daily total here; the local page shows it next to the site's own
consumption.

Two ways to get watts, and the record always says which was used:

* ``estimated``: a straight line between the board's idle and full-load
  power, by CPU load, plus a fixed allowance for the modem and adapters.
  The defaults are published figures for a Raspberry Pi 5, not measurements
  of your box.
* ``measured``: set ``BOX_MEASURED_WATTS`` to the reading from a USB power
  meter on the box's supply. This replaces the estimate entirely.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KEEP_DAYS = 35


@dataclass(frozen=True)
class PowerModel:
    idle_watts: float = 3.0
    full_watts: float = 8.8
    extra_watts: float = 1.0  # modem, serial adapters, SSD
    measured_watts: float | None = None

    @classmethod
    def from_env(cls) -> PowerModel:
        measured = os.environ.get("BOX_MEASURED_WATTS", "").strip()
        return cls(
            idle_watts=float(os.environ.get("BOX_IDLE_WATTS", "3.0")),
            full_watts=float(os.environ.get("BOX_FULL_WATTS", "8.8")),
            extra_watts=float(os.environ.get("BOX_EXTRA_WATTS", "1.0")),
            measured_watts=float(measured) if measured else None,
        )

    @property
    def method(self) -> str:
        return "measured" if self.measured_watts is not None else "estimated"

    def watts(self, cpu_fraction: float) -> float:
        if self.measured_watts is not None:
            return self.measured_watts
        load = min(max(cpu_fraction, 0.0), 1.0)
        return self.idle_watts + load * (self.full_watts - self.idle_watts) + self.extra_watts


class EnergyLedger:
    """Watt-hours used by the box per local day, kept in a small JSON file."""

    def __init__(self, path: str | Path, model: PowerModel | None = None,
                 utc_offset_hours: float = 1.0) -> None:
        self.path = Path(path)
        self.model = model or PowerModel()
        self.utc_offset_hours = utc_offset_hours
        self.days: dict[str, dict[str, float]] = read_ledger(self.path).get("days", {})

    def _day(self, now: float) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(now + self.utc_offset_hours * 3600))

    def add(self, cpu_fraction: float, seconds: float, now: float | None = None) -> float:
        """Record ``seconds`` at the given CPU load. Returns the watts used."""
        now = time.time() if now is None else now
        watts = self.model.watts(cpu_fraction)
        day = self.days.setdefault(self._day(now), {"wh": 0.0, "seconds": 0.0, "cpu_seconds": 0.0})
        day["wh"] += watts * seconds / 3600
        day["seconds"] += seconds
        day["cpu_seconds"] += min(max(cpu_fraction, 0.0), 1.0) * seconds
        for old in sorted(self.days)[:-KEEP_DAYS]:
            del self.days[old]
        self._save(now)
        return watts

    def _save(self, now: float) -> None:
        payload = {"method": self.model.method, "updated_at": now, "days": self.days,
                   "power_model": {"idle_watts": self.model.idle_watts,
                                   "full_watts": self.model.full_watts,
                                   "extra_watts": self.model.extra_watts,
                                   "measured_watts": self.model.measured_watts}}
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, self.path)
        except OSError:
            pass  # a full or read-only disk must not stop the watchdog


def read_ledger(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def summarise(ledger: dict[str, Any], site_wh_last_day: float | None = None) -> dict[str, Any]:
    """Figures for the local page from a ledger file's contents.

    Uses the most recent day with at least an hour recorded, scaled to a full
    day at that day's average power, so a box switched on at noon is not
    reported as using half of what it does.
    """
    days = {d: v for d, v in (ledger.get("days") or {}).items() if v.get("seconds", 0) >= 3600}
    if not days:
        return {"available": False}
    latest = max(days)
    rec = days[latest]
    watts = rec["wh"] * 3600 / rec["seconds"]
    out: dict[str, Any] = {
        "available": True,
        "method": ledger.get("method", "estimated"),
        "day": latest,
        "average_watts": round(watts, 2),
        "wh_per_day": round(watts * 24, 1),
        "average_cpu_pct": round(100 * rec.get("cpu_seconds", 0.0) / rec["seconds"], 1),
        "days_recorded": len(days),
    }
    if site_wh_last_day and site_wh_last_day > 0:
        out["site_wh_last_day"] = round(site_wh_last_day)
        out["share_of_site_pct"] = round(100 * watts * 24 / site_wh_last_day, 2)
    return out
