"""On-site learning: models train on the box from the site's own history.

Uses the simulated site in inference/sitesim.py with injected faults.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xgboost")

from inference.sitelearn import SiteLearner, slot_series, time_features  # noqa: E402
from inference.sitesim import DAY_S, T0, Faults, SimHistory, SimSite  # noqa: E402


def run(tmp: Path, faults: Faults, days: int, check_days: list[float], hours: list[float] | None = None,
        seed: int = 0, sign: float = 1.0) -> dict[float, set[str]]:
    data = SimSite(days=days, seed=seed, faults=faults).run()
    if sign != 1.0:
        ts, p = data["battery.power_w"]
        data["battery.power_w"] = (ts, sign * p)
    h = SimHistory(data)
    sl = SiteLearner(h, tmp)
    out: dict[float, set[str]] = {}
    times = sorted({T0 + d * DAY_S for d in range(7, days, 1)} |
                   {T0 + d * DAY_S + hr * 3600 for d in check_days for hr in (hours or [12])})
    for t in times:
        h.now = float(t)
        codes = {f["code"] for f in sl.step(float(t))}
        out[round((t - T0) / DAY_S, 3)] = codes
    return out


def all_codes(result: dict[float, set[str]], after: float = 0, before: float = 1e9) -> set[str]:
    return set().union(*[c for d, c in result.items() if after <= d < before]) if result else set()


def test_slot_series_and_time_features() -> None:
    rows = [{"timestamp": (T0 + i * 60) * 1000, "value": float(i), "quality": 0} for i in range(30)]
    rows.append({"timestamp": (T0 + 31 * 60) * 1000, "value": 999.0, "quality": 4})  # bad quality
    ts, v = slot_series(rows)
    assert list(ts) == [T0, T0 + 900] and v[0] == pytest.approx(7.0) and v[1] == pytest.approx(22.0)
    f = time_features(np.array([T0 - 3600]), 1.0)  # 00:00 WAT, a Sunday
    assert f[0, 0] == pytest.approx(0.0, abs=1e-9) and f[0, 1] == pytest.approx(1.0) and f[0, 4] == 1.0


def test_clean_site_raises_nothing(tmp_path: Path) -> None:
    result = run(tmp_path, Faults(), days=40, check_days=[])
    assert all_codes(result) == set()


def test_no_models_before_enough_history(tmp_path: Path) -> None:
    h = SimHistory(SimSite(days=5, seed=1).run())
    sl = SiteLearner(h, tmp_path)
    assert sl.step(h.now) == [] and sl.status() == {}


def test_dirty_panels_detected_within_three_days(tmp_path: Path) -> None:
    result = run(tmp_path, Faults(soiling_day=25), days=33, check_days=[])
    assert "pv_underperform" not in all_codes(result, before=25)
    hits = [d for d, c in result.items() if "pv_underperform" in c]
    assert hits and min(hits) - 25 <= 3
    assert "battery_degrading" not in all_codes(result)


def test_battery_resistance_rise_detected(tmp_path: Path) -> None:
    result = run(tmp_path, Faults(resistance_day=25), days=33, check_days=[])
    assert "battery_degrading" not in all_codes(result, before=25)
    hits = [d for d, c in result.items() if "battery_degrading" in c]
    assert hits and min(hits) - 25 <= 6


def test_battery_sign_convention_is_learned(tmp_path: Path) -> None:
    """Some inverters report discharge as positive power, others as negative."""
    result = run(tmp_path, Faults(resistance_day=25), days=33, check_days=[], sign=-1.0)
    assert any("battery_degrading" in c for d, c in result.items() if d >= 25)


def test_night_theft_detected_same_night(tmp_path: Path) -> None:
    result = run(tmp_path, Faults(theft_day=20), days=22, check_days=[19, 20],
                 hours=[1, 2, 3, 4, 4.5])
    assert "load_high" not in all_codes(result, before=20)
    assert "load_high" in all_codes(result, after=20, before=20.3)


def test_models_persist_and_reset(tmp_path: Path) -> None:
    h = SimHistory(SimSite(days=16, seed=2).run())
    sl = SiteLearner(h, tmp_path)
    sl.step(h.now)
    assert {"pv:inv1", "battery:inv1", "load:inv1"} <= set(sl.status())
    again = SiteLearner(h, tmp_path)
    assert "battery:inv1" in again.status()
    again.reset("battery:")
    assert "battery:inv1" not in SiteLearner(h, tmp_path).status()


def test_findings_become_sms_alerts(tmp_path: Path) -> None:
    from sms.service import Contact, SmsConfig, SmsService
    from sms.store import SmsStore

    sent: list[tuple[str, str]] = []

    class Backend:
        def send(self, phone: str, text: str) -> bool:
            sent.append((phone, text))
            return True

        def receive(self) -> list[tuple[str, str]]:
            return []

    cfg = SmsConfig(site_name="Kuje", technicians=[Contact("+2348030000001")], quiet_hours=None,
                    daily_summary_hour=None, hold_s=0)
    svc = SmsService(cfg, Backend(), SmsStore(str(tmp_path / "s.db")))
    finding = {"code": "pv_underperform", "asset_id": "inv1", "severity": "warning",
               "values": {"pct": 61, "days": 5}}
    svc.on_findings({"at": T0, "findings": [finding]})
    svc.tick(T0 + 60)
    assert sent and "Solar panels giving 61%" in sent[0][1]
    svc.tick(T0 + 4000)  # findings older than an hour are ignored, so the fault clears
    assert len(sent) == 1


def test_trains_from_the_real_edge_database(edge_db, tmp_path: Path) -> None:
    """End to end on the box's own SQLite store, not the simulator's history."""
    import time

    data = SimSite(days=16, seed=3).run()
    shift = time.time() - max(ts[-1] for ts, _ in data.values())
    rows = []
    for metric, (ts, vals) in data.items():
        rows += [(int((t + shift) * 1000), "inv1", metric, float(v), 0) for t, v in zip(ts, vals, strict=True)]
    edge_db.insert_measurements(rows)
    sl = SiteLearner(edge_db, tmp_path)
    assert sl.step() == []
    assert {"pv:inv1", "battery:inv1", "load:inv1"} <= set(sl.status())


def test_inverters_reporting_per_tracker_and_load_power(tmp_path: Path) -> None:
    """Felicity-style metrics: solar.mppt1_power_w and load.power_w, no totals."""
    data = SimSite(days=16, seed=4).run()
    data["solar.mppt1_power_w"] = data.pop("solar.pv_power_w")
    data["load.power_w"] = data.pop("system.ac_consumption_w")
    h = SimHistory(data)
    sl = SiteLearner(h, tmp_path)
    sl.step(h.now)
    assert {"pv:inv1", "battery:inv1", "load:inv1"} <= set(sl.status())
