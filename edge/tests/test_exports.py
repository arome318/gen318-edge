"""The monthly operating record and the open fault dataset export."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from common.operating_record import SCHEMA, build_record, checksum, daily_csv, month_bounds, verify
from sms.dataset import COLUMNS, build_rows, summary, to_csv
from sms.store import SmsStore

REPO = Path(__file__).resolve().parents[2]
TECH = "+2348030000001"
MONTH = "2026-03"
START, END, DAYS = month_bounds(MONTH, 1.0)        # local midnight 1 March, WAT


def seed_ten_days(db, outage: bool = True) -> None:
    """Ten days from the start of March: 2 kW load, two 500 W trackers, one 3-hour outage on day 2."""
    rows = []
    for i in range(10 * 288):                       # a reading every 5 minutes
        t = START + i * 300
        off = outage and START + 86400 + 3 * 3600 <= t < START + 86400 + 6 * 3600
        ms = int(t * 1000)
        rows += [
            (ms, "inv1", "load.power_w", 2000.0, 0),
            (ms, "inv1", "solar.mppt1_power_w", 500.0, 0),
            (ms, "inv1", "solar.mppt2_power_w", 500.0, 0),
            (ms, "inv1", "inverter.ac_output_voltage_v", 0.0 if off else 230.0, 0),
            (ms, "inv1", "battery.soc_pct", 12.0 if off else 70.0, 0),
        ]
    db.insert_measurements(rows)


@pytest.fixture
def store(tmp_path: Path) -> SmsStore:
    s = SmsStore(str(tmp_path / "sms.db"))
    raised = START + 86400 + 3 * 3600 + 120
    s.record_alert("AAAA", "power_off", "critical", "inv1", {"voltage": 0}, raised)
    s.record_delivery("AAAA", TECH, "technician", "alert", True, raised, "Kuje: Power is OFF")
    s.clear_alert("power_off", "inv1", raised + 3 * 3600)
    s.record_label("AAAA", 1, "Musa says breaker tripped", TECH, raised + 900)
    s.record_alert("BBBB", "battery_low", "warning", "inv1", {"soc": 15}, START + 5 * 86400)
    s.record_label("BBBB", 2, "", TECH, START + 5 * 86400 + 600)
    s.record_alert("CCCC", "inverter_hot", "warning", "inv1", {"temp": 72}, START + 6 * 86400)   # never labelled
    return s


# ── operating record ───────────────────────────────────────────


def test_month_bounds() -> None:
    assert DAYS == 31 and END - START == 31 * 86400
    assert month_bounds("2028-02", 0)[2] == 29
    for bad in ("2026-13", "1999-01"):
        with pytest.raises(ValueError):
            month_bounds(bad, 1.0)


def test_record_reports_only_what_the_box_saw(edge_db, store) -> None:
    seed_ten_days(edge_db)
    r = build_record(edge_db, store, MONTH, site_name="Kuje", device_id="box-1",
                     register_map_verification="community", now=END + 3600)
    assert r["schema"] == SCHEMA and r["month_complete"] is True
    t = r["totals"]
    assert t["data_coverage_pct"] == pytest.approx(100 * 10 / 31, abs=0.1) and t["days_with_data"] == 10
    assert t["energy_delivered_kwh"] == pytest.approx(2.0 * 240, rel=1e-3)      # 2 kW for 240 hours
    assert t["solar_kwh"] == pytest.approx(1.0 * 240, rel=1e-3)                 # the two trackers added up
    assert t["genset_kwh"] is None                                              # no genset: unknown, not zero
    assert t["outage_hours"] == 3.0 and t["power_available_pct"] == pytest.approx(100 * 237 / 240, abs=0.01)
    assert t["lowest_battery_pct"] == 12.0
    assert t["alerts"] == {"critical": 1, "warning": 2}
    assert t["alerts_labelled"] == {"real": 1, "false_alarm": 1, "other_cause": 0}
    assert t["sms_sent"] == {"alert": 1}

    day2, day11 = r["daily"][1], r["daily"][10]
    assert day2["date"] == "2026-03-02" and day2["outage_minutes"] == 180 and day2["alerts_critical"] == 1
    assert day2["power_available_pct"] == 87.5 and day2["energy_delivered_kwh"] == 48.0
    # a day the box did not see is unknown, not "power on" and not "power off"
    assert day11["data_coverage_pct"] == 0.0 and day11["power_available_pct"] is None
    assert day11["outage_minutes"] is None and day11["energy_delivered_kwh"] is None

    alert = next(a for a in r["alerts"] if a["code"] == "power_off")
    assert alert["minutes_open"] == 180.0 and alert["technician_label"] == "real"
    assert "Musa" not in json.dumps(r) and TECH not in json.dumps(r)             # no notes, no numbers


def test_record_checksum_detects_changes(edge_db, store) -> None:
    seed_ten_days(edge_db)
    r = build_record(edge_db, store, MONTH, site_name="Kuje", device_id="box-1", now=END + 1)
    assert verify(r) and r["record_sha256"] == checksum(r)
    reloaded = json.loads(json.dumps(r))
    assert verify(reloaded)
    reloaded["totals"]["power_available_pct"] = 100.0
    assert not verify(reloaded)


def test_record_part_way_through_the_month(edge_db) -> None:
    seed_ten_days(edge_db, outage=False)
    r = build_record(edge_db, None, MONTH, site_name="Kuje", device_id="box-1", now=START + 10 * 86400)
    assert r["month_complete"] is False
    assert r["totals"]["data_coverage_pct"] == 100.0        # of the ten days that have happened
    assert r["totals"]["power_available_pct"] == 100.0 and r["alerts"] == []


def test_record_with_no_data_says_so(edge_db) -> None:
    r = build_record(edge_db, None, MONTH, site_name="Kuje", device_id="box-1", now=END + 1)
    t = r["totals"]
    assert t["data_coverage_pct"] == 0.0 and t["power_available_pct"] is None and t["energy_delivered_kwh"] is None
    csv = daily_csv(r).splitlines()
    assert csv[0].startswith("date,data_coverage_pct,power_available_pct") and len(csv) == 32
    assert csv[1] == "2026-03-01,0.0,,,,,,,0,0"


def test_operating_record_tool_writes_and_checks_files(edge_db, store, tmp_path: Path) -> None:
    seed_ten_days(edge_db)
    cfg = tmp_path / "sms.yaml"
    cfg.write_text("site_name: Kuje Mini-grid\nutc_offset_hours: 1\n")
    tool = [sys.executable, str(REPO / "tools" / "operating_record.py")]
    out = subprocess.run(
        [*tool, MONTH, "--out", str(tmp_path / "records"), "--db", edge_db._db_path,
         "--sms-db", str(tmp_path / "sms.db"), "--sms-config", str(cfg), "--adapters", str(tmp_path / "none.yaml")],
        capture_output=True, text=True, check=True).stdout
    assert "energy delivered 480.0 kWh" in out and "register maps verified to: unknown" in out
    path = tmp_path / "records" / "operating-record-kuje-mini-grid-2026-03.json"
    assert json.loads(path.read_text())["site_name"] == "Kuje Mini-grid" and path.with_suffix(".csv").exists()
    assert "unchanged" in subprocess.run([*tool, "--check", str(path)], capture_output=True, text=True).stdout
    path.write_text(path.read_text().replace('"outage_hours": 3.0', '"outage_hours": 0.0'))
    changed = subprocess.run([*tool, "--check", str(path)], capture_output=True, text=True)
    assert changed.returncode == 1 and "CHANGED" in changed.stdout


# ── fault dataset ──────────────────────────────────────────────


def test_dataset_rows_carry_labels_and_context_but_nothing_identifying(edge_db, store) -> None:
    seed_ten_days(edge_db)
    rows = build_rows(store, edge_db, "box-1")
    assert [r["code"] for r in rows] == ["power_off", "battery_low", "inverter_hot"]
    off, low, hot = rows
    assert off["label"] == "real" and off["minutes_to_label"] == 15.0 and off["minutes_open"] == 180.0
    assert low["label"] == "false_alarm" and low["values"] == {"soc": 15}
    assert hot["label"] is None and hot["minutes_to_label"] is None
    assert (off["month"], off["weekday"], off["hour"]) == ("2026-03", 0, 3)       # Monday 2 March, 03:02 WAT
    # the hour before battery_low: steady site, both trackers added up
    assert low["before"]["battery_soc_pct"] == 70.0 and low["before"]["solar_power_w"] == 1000.0
    assert low["before"]["load_power_w"] == 2000.0 and low["before"]["genset_power_w"] is None

    text = json.dumps(rows)
    for secret in ("Musa", TECH, "box-1", "inv1", "AAAA", "raised_at", "Kuje"):
        assert secret not in text
    assert len(off["site"]) == 12 and {r["site"] for r in rows} == {off["site"]}


def test_dataset_site_id_is_stable_and_differs_between_boxes(store, tmp_path: Path) -> None:
    again = SmsStore(str(tmp_path / "sms.db"))
    assert again.site_pseudonym("box-1") == store.site_pseudonym("box-1") != store.site_pseudonym("box-2")
    assert SmsStore(str(tmp_path / "other.db")).site_pseudonym("box-1") != store.site_pseudonym("box-1")


def test_dataset_summary_csv_and_empty_export(store, tmp_path: Path) -> None:
    rows = build_rows(store, None, "box-1")
    info = summary(rows)
    assert (info["alerts"], info["labelled"], info["sites"]) == (3, 2, 1)
    assert info["labels"] == {"real": 1, "false_alarm": 1, "other_cause": 0}
    assert info["by_code"]["battery_low"] == {"alerts": 1, "real": 0, "false_alarm": 1, "other_cause": 0}
    assert [r["code"] for r in build_rows(store, None, "box-1", labelled_only=True)] == ["power_off", "battery_low"]
    lines = to_csv(rows).splitlines()
    assert lines[0].split(",") == list(COLUMNS) and len(lines) == 4
    assert all(len(line.split(",")) == len(COLUMNS) for line in lines)

    empty = SmsStore(str(tmp_path / "empty.db"))
    assert build_rows(empty, None, "box-1") == [] and summary([])["alerts"] == 0
    tool = subprocess.run(
        [sys.executable, str(REPO / "tools" / "export_fault_dataset.py"), "--out", str(tmp_path / "ds"),
         "--sms-db", str(tmp_path / "empty.db"), "--db", str(tmp_path / "none.db"),
         "--sms-config", str(tmp_path / "none.yaml")], capture_output=True, text=True, check=True).stdout
    assert "0 alerts, 0 labelled" in tool and "No technician labels yet" in tool
    assert (tmp_path / "ds" / "faults.csv").read_text().count("\n") == 1
