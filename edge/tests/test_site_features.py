"""Curtailment-aware solar, crash-safe learned state, run-out warning, label-driven
tuning, standalone retention and the box's own energy ledger."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xgboost")

from common.energy import EnergyLedger, PowerModel, read_ledger, summarise  # noqa: E402
from inference.sitelearn import SiteLearnConfig, SiteLearner, tuned_config  # noqa: E402
from inference.sitesim import DAY_S, T0, Faults, SimHistory, SimSite  # noqa: E402
from sms.commands import parse  # noqa: E402
from sms.rules import Thresholds  # noqa: E402
from sms.service import Contact, SmsConfig, SmsService  # noqa: E402
from sms.store import SmsStore  # noqa: E402
from sms.tuning import decide, steps_from_labels, tuned_thresholds  # noqa: E402

TECH = "+2348030000001"


def codes_over(site: SimSite, tmp: Path, start_day: int, step_h: float = 6,
               cfg: SiteLearnConfig | None = None) -> dict[float, set[str]]:
    h = SimHistory(site.run())
    sl = SiteLearner(h, tmp, cfg)
    out = {}
    for t in np.arange(T0 + start_day * DAY_S, T0 + site.days * DAY_S, step_h * 3600):
        h.now = float(t)
        out[(t - T0) / DAY_S] = {f["code"] for f in sl.step(float(t))}
    return out


# ── solar: a full bank is not a dirty panel ────────────────────


def test_simulator_throttles_panels_when_the_bank_is_full() -> None:
    data = SimSite(days=12, seed=0).run()
    _, pv = data["solar.pv_power_w"]
    _, soc = data["battery.soc_pct"]
    _, load = data["system.ac_consumption_w"]
    full = soc[:-1] >= 99  # full at the start of the step
    assert full.sum() > 50
    assert np.all(pv[1:][full] <= load[1:][full] + 1e-6)


def test_load_drop_with_healthy_panels_is_not_reported_as_dirty_panels(tmp_path: Path) -> None:
    """Daytime customers leave on day 25: the bank fills early and the inverter
    throttles the panels. Before the curtailment mask this raised pv_underperform."""
    site = SimSite(days=40, seed=0, load_scale_from=(25, 0.5))
    result = codes_over(site, tmp_path, start_day=7)
    assert not any("pv_underperform" in c for c in result.values())


def test_dirty_panels_still_detected_with_curtailment(tmp_path: Path) -> None:
    site = SimSite(days=33, seed=0, faults=Faults(soiling_day=25))
    result = codes_over(site, tmp_path, start_day=7)
    hits = [d for d, c in result.items() if "pv_underperform" in c]
    assert hits and 25 <= min(hits) <= 29


def test_solar_model_records_how_much_was_curtailed(tmp_path: Path) -> None:
    h = SimHistory(SimSite(days=12, seed=1).run())
    sl = SiteLearner(h, tmp_path)
    sl.step(h.now)
    info = sl.status()["pv:inv1"]
    assert info["curtailed_pct"] > 0 and info["bytes"] > 0 and info["fit_s"] >= 0


# ── learned state survives power cuts ──────────────────────────


def test_truncated_meta_file_does_not_stop_the_service(tmp_path: Path) -> None:
    h = SimHistory(SimSite(days=16, seed=2).run())
    SiteLearner(h, tmp_path).step(h.now)
    meta = tmp_path / "site_learning" / "meta.json"
    meta.write_text("")  # what a power cut during an in-place write leaves behind
    sl = SiteLearner(h, tmp_path)
    assert sl.status() == {}
    sl.step(h.now)  # relearns from the readings still on the box
    assert {"pv:inv1", "battery:inv1", "load:inv1"} <= set(sl.status())
    assert json.loads(meta.read_text())
    assert not list((tmp_path / "site_learning").glob("*.tmp"))


def test_damaged_model_file_is_relearned(tmp_path: Path) -> None:
    h = SimHistory(SimSite(days=16, seed=2).run())
    SiteLearner(h, tmp_path).step(h.now)
    for name in ("pv__inv1.ubj", "load__inv1__mid.ubj", "battery__inv1.ubj"):
        (tmp_path / "site_learning" / name).write_bytes(b"\x00garbage")
    sl = SiteLearner(h, tmp_path)
    assert sl.step(h.now) == []
    sl.step(h.now)
    assert {"pv:inv1", "battery:inv1", "load:inv1"} <= set(sl.status())


def test_one_failing_check_does_not_silence_the_others(tmp_path: Path) -> None:
    h = SimHistory(SimSite(days=12, seed=0).run())
    sl = SiteLearner(h, tmp_path)

    def boom(asset: str, now: float) -> None:
        raise RuntimeError("bad asset")

    sl._pv = boom  # type: ignore[method-assign]
    sl._load_check = lambda asset, now: {"code": "load_high", "asset_id": asset,  # type: ignore[method-assign]
                                         "severity": "warning", "values": {}}
    assert [f["code"] for f in sl.step(h.now)] == ["load_high"]


# ── run-out warning ────────────────────────────────────────────


def test_runout_learns_the_bank_size_from_readings(tmp_path: Path) -> None:
    h = SimHistory(SimSite(days=12, seed=0, bank_kwh=18).run())
    sl = SiteLearner(h, tmp_path)
    sl.step(h.now)
    assert sl.status()["runout:inv1"]["usable_kwh_estimate"] == pytest.approx(18, rel=0.05)


def test_runout_warns_hours_before_the_power_goes_off(tmp_path: Path) -> None:
    """A 17 kWh bank on this load lasts most nights and fails after a cloudy day."""
    data = SimSite(days=60, seed=0, bank_kwh=17).run()
    ts, soc = data["battery.soc_pct"]
    _, out_v = data["inverter.ac_output_voltage_v"]
    h = SimHistory(data)
    sl = SiteLearner(h, tmp_path)
    sl._pv = sl._battery = sl._load_check = lambda asset, now: None  # type: ignore[method-assign]
    warned = []
    for t in np.arange(T0 + 9 * DAY_S, T0 + 60 * DAY_S, 1800):
        h.now = float(t)
        if any(f["code"] == "battery_runout" for f in sl.step(float(t))):
            warned.append(float(t))
    assert warned
    # every warning is followed by a bank that really did get close to empty
    for w in warned:
        assert soc[(ts >= w) & (ts <= w + 8 * 3600)].min() <= 15
    off = out_v < 150
    cuts = [c for c in ts[1:][off[1:] & ~off[:-1]] if c > T0 + 9 * DAY_S]
    leads = [c - min(w for w in warned if c - 8 * 3600 <= w < c)
             for c in cuts if any(c - 8 * 3600 <= w < c for w in warned)]
    assert leads and min(leads) >= 2 * 3600


def test_runout_stays_quiet_on_a_site_that_runs_out_most_nights(tmp_path: Path) -> None:
    """A 15 kWh bank on this load empties nearly every night: a sizing problem
    the technician knows about, not something to text about nightly."""
    site = SimSite(days=20, seed=0)
    result = codes_over(site, tmp_path, start_day=14, step_h=1)
    assert not any("battery_runout" in c for c in result.values())


def test_runout_is_silent_without_fresh_readings(tmp_path: Path) -> None:
    h = SimHistory(SimSite(days=12, seed=0, bank_kwh=18).run())
    sl = SiteLearner(h, tmp_path)
    sl.step(h.now)
    assert sl._runout("inv1", h.now + 3 * 3600) is None


def test_runout_warning_does_not_wait_for_morning(tmp_path: Path) -> None:
    sent: list[str] = []

    class Backend:
        def send(self, phone: str, text: str) -> bool:
            sent.append(text)
            return True

        def receive(self) -> list[tuple[str, str]]:
            return []

    cfg = SmsConfig(site_name="Kuje", technicians=[Contact(TECH)], quiet_hours=(22, 6),
                    daily_summary_hour=None, hold_s=0)
    svc = SmsService(cfg, Backend(), SmsStore(str(tmp_path / "s.db")))
    night = T0 + 22.5 * 3600  # 23:30 WAT
    svc.on_findings({"at": night, "findings": [
        {"code": "battery_runout", "asset_id": "inv1", "values": {"at": "02:15", "hours": 2.8, "soc": 31}},
        {"code": "pv_underperform", "asset_id": "inv1", "values": {"pct": 60, "days": 5}},
    ]})
    svc.tick(night + 30)
    assert len(sent) == 1 and "run out around 02:15" in sent[0]


# ── technician replies tune sensitivity ────────────────────────


def test_steps_from_labels() -> None:
    assert steps_from_labels([2, 2]) == 0
    assert steps_from_labels([2, 2, 2]) == 1
    assert steps_from_labels([2, 3, 2, 3, 2]) == 1          # "other cause" is not counted
    assert steps_from_labels([2, 2, 2, 2, 2, 1]) == 2       # the old "real" is outside the window
    assert steps_from_labels([1, 2, 2, 2, 2]) == 0          # a recent "real" undoes it
    assert decide({"power_off": [2, 2, 2, 2, 2], "battery_low": [2, 2, 2]}) == {"battery_low": 1}


def test_tuned_thresholds_are_bounded() -> None:
    base = Thresholds(soc_warning_pct=20, soc_critical_pct=17, overload_ratio=0.98)
    t = tuned_thresholds(base, {"battery_low": 9, "overload": 2, "inverter_hot": 1})
    assert t.soc_warning_pct == 19 and t.overload_ratio == 1.0 and t.inverter_temp_c == 73
    assert t.soc_critical_pct == base.soc_critical_pct and t.battery_temp_c == base.battery_temp_c
    c = tuned_config(SiteLearnConfig(), {"pv_underperform": 5, "load_high": 1, "bogus": 2})
    assert c.pv_alert_ratio == pytest.approx(0.65) and c.load_high_factor == pytest.approx(1.45)


class Backend:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, phone: str, text: str) -> bool:
        self.sent.append((phone, text))
        return True

    def receive(self) -> list[tuple[str, str]]:
        return []


def make_service(tmp: Path, published: list[dict] | None = None) -> tuple[SmsService, Backend]:
    cfg = SmsConfig(site_name="Kuje", technicians=[Contact(TECH)], quiet_hours=None,
                    daily_summary_hour=None, hold_s=0, realert_s=10**9)
    backend = Backend()
    svc = SmsService(cfg, backend, SmsStore(str(tmp / "sms.db")),
                     publish_tuning=published.append if published is not None else None)
    return svc, backend


def raise_and_label(svc: SmsService, soc: float, label: str, t: float) -> None:
    svc.on_telemetry({"timestamp": t * 1000, "asset_id": "inv1", "metric_name": "battery.soc_pct", "value": soc})
    svc.tick(t + 1)
    svc.handle_inbound(TECH, label, t + 2)
    svc.on_telemetry({"timestamp": (t + 3) * 1000, "asset_id": "inv1", "metric_name": "battery.soc_pct", "value": 80})
    svc.tick(t + 4)


def test_three_false_alarm_replies_loosen_the_warning_and_a_real_one_restores_it(tmp_path: Path) -> None:
    svc, backend = make_service(tmp_path)
    for i in range(3):
        raise_and_label(svc, 19, "2", T0 + i * 100)
    assert svc.tuning == {"battery_low": 1} and svc.thresholds.soc_warning_pct == 17.5
    assert any("less sensitive (step 1 of 2)" in text for _, text in backend.sent)

    before = len(backend.sent)
    svc.on_telemetry({"timestamp": (T0 + 500) * 1000, "asset_id": "inv1", "metric_name": "battery.soc_pct", "value": 19})
    svc.tick(T0 + 501)
    assert len(backend.sent) == before  # 19% no longer alerts

    raise_and_label(svc, 15, "1", T0 + 600)  # a real one at 15%
    assert svc.tuning == {} and svc.thresholds.soc_warning_pct == 20
    assert any("back to the installer's setting" in text for _, text in backend.sent)


def test_tuning_survives_a_restart_and_reset_clears_it(tmp_path: Path) -> None:
    svc, _ = make_service(tmp_path)
    for i in range(3):
        raise_and_label(svc, 19, "2", T0 + i * 100)
    again, _ = make_service(tmp_path)
    assert again.tuning == {"battery_low": 1}
    assert parse("reset tuning").kind == "reset" and parse("RESET").kind == "reset"
    assert again.handle_inbound("+2348039999999", "RESET TUNING", T0 + 900) is None  # not a technician
    assert again.tuning == {"battery_low": 1}
    assert "installer" in (again.handle_inbound(TECH, "RESET TUNING", T0 + 901) or "")
    assert again.tuning == {} and make_service(tmp_path)[0].tuning == {}


def test_critical_alerts_are_never_loosened(tmp_path: Path) -> None:
    svc, _ = make_service(tmp_path)
    for i in range(5):
        raise_and_label(svc, 5, "2", T0 + i * 100)  # battery_critical
    assert svc.tuning == {} and svc.thresholds == svc.cfg.thresholds


def test_learned_warning_steps_reach_the_models(tmp_path: Path) -> None:
    published: list[dict] = []
    svc, _ = make_service(tmp_path, published)
    for i in range(3):
        t = T0 + i * 4000
        svc.on_findings({"at": t, "findings": [{"code": "load_high", "asset_id": "inv1",
                                                "values": {"actual_w": 2000, "expected_w": 900}}]})
        svc.tick(t + 1)
        svc.handle_inbound(TECH, "2", t + 2)
        svc.tick(t + 3700)  # finding goes stale, alert clears
    svc.tick(T0 + 20000)
    assert published and published[-1]["steps"] == {"load_high": 1}
    sl = SiteLearner(SimHistory(SimSite(days=1).run()), tmp_path)
    sl.set_tuning(published[-1]["steps"])
    assert sl.cfg.load_high_factor == pytest.approx(1.45) and sl.base_cfg.load_high_factor == 1.3
    sl.set_tuning({})
    assert sl.cfg == sl.base_cfg


def test_auto_tune_can_be_switched_off(tmp_path: Path) -> None:
    cfg = SmsConfig.from_yaml("auto_tune: false\ntechnicians: [{phone: '08030000001'}]\nquiet_hours: null\nhold_s: 0")
    svc = SmsService(cfg, Backend(), SmsStore(str(tmp_path / "s.db")))
    for i in range(3):
        raise_and_label(svc, 19, "2", T0 + i * 100)
    assert svc.tuning == {} and cfg.urgent_warnings == ["battery_runout"]


# ── standalone retention ───────────────────────────────────────


def test_standalone_box_prunes_old_history_and_a_syncing_box_does_not(edge_db, edge_config) -> None:
    from dataclasses import replace

    from collector.main import CollectorService

    now = time.time()
    old, recent = int((now - 60 * 86400) * 1000), int((now - 5 * 86400) * 1000)
    edge_db.insert_measurements([(old, "a", "m", 1.0, 0), (recent, "a", "m", 2.0, 0)])
    edge_db.insert_event("e-old", "x", "info", {}, old)
    edge_db.enqueue_sync(priority=3, payload={"k": 1})

    def service(url: str) -> CollectorService:
        svc = CollectorService.__new__(CollectorService)
        svc.config = replace(edge_config, cloud_api_url=url, local_retention_days=45)
        svc.db = edge_db
        svc._last_local_prune_at = 0.0
        return svc

    service("https://operator.example")._prune_standalone(now)
    assert edge_db.count_unsynced_measurements() == 2  # the only copy: kept

    service("")._prune_standalone(now)
    assert edge_db.count_unsynced_measurements() == 1
    assert edge_db.get_events(limit=10) == []
    assert edge_db.get_queue_depth_by_priority()  # a fresh queue item is younger than 45 days


# ── the box's own energy ───────────────────────────────────────


def test_power_model_and_ledger(tmp_path: Path) -> None:
    model = PowerModel(idle_watts=3, full_watts=9, extra_watts=1)
    assert model.watts(0) == 4 and model.watts(0.5) == 7 and model.watts(7) == 10
    assert PowerModel(measured_watts=5.2).watts(0.9) == 5.2

    path = tmp_path / "metrics" / "energy.json"
    ledger = EnergyLedger(path, model)
    for i in range(120):  # two hours at 50% CPU, one sample a minute
        ledger.add(0.5, 60, now=T0 + i * 60)
    summary = summarise(read_ledger(path), site_wh_last_day=40_000)
    assert summary["method"] == "estimated" and summary["average_watts"] == 7.0
    assert summary["wh_per_day"] == 168.0 and summary["share_of_site_pct"] == 0.42
    assert EnergyLedger(path, model).days == ledger.days  # survives a restart

    assert summarise({}) == {"available": False}
    path.write_text("{not json")
    assert read_ledger(path) == {}


def test_ledger_keeps_a_bounded_number_of_days(tmp_path: Path) -> None:
    ledger = EnergyLedger(tmp_path / "e.json", PowerModel(measured_watts=5))
    for d in range(50):
        ledger.add(0.1, 3600, now=T0 + d * DAY_S)
    assert len(ledger.days) == 35 and read_ledger(tmp_path / "e.json")["method"] == "measured"
