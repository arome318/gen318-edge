"""Write a month's operating record for this site.

    python tools/operating_record.py 2026-11 --out records/

Run it on the box (or anywhere with the box's two database files). It
writes two files and sends nothing anywhere:

    operating-record-<site>-2026-11.json   everything, with a checksum
    operating-record-<site>-2026-11.csv    one row per day, for a spreadsheet

    python tools/operating_record.py --check records/operating-record-kuje-2026-11.json

checks that a record has not been changed since it was made.

The format is described in docs/OPERATING_RECORD.md. Readings older than
the box's retention (45 days by default) are gone, so make each month's
record within a couple of weeks of the month ending.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "edge"))

from common.operating_record import build_record, daily_csv, verify  # noqa: E402

RANK = {"community": 0, "bench": 1, "field": 2}


def map_verification(adapter_config: str) -> str:
    """The weakest verification level among the register maps this site reads through."""
    try:
        import yaml

        from collector.adapters.vendor_map import load_vendor_map

        raw = yaml.safe_load(Path(adapter_config).read_text()) or {}
        levels = [load_vendor_map(a["config"]["vendor_map"]).verification
                  for a in raw.get("adapters", [])
                  if a.get("enabled", True) and isinstance(a.get("config"), dict) and a["config"].get("vendor_map")]
    except Exception:  # noqa: BLE001 - a missing or odd config must not stop the record
        return "unknown"
    return min(levels, key=lambda v: RANK.get(str(v), -1)) if levels else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a month's operating record for this site.")
    ap.add_argument("month", nargs="?", help="YYYY-MM, in the site's local time")
    ap.add_argument("--out", default=".", help="folder to write into")
    ap.add_argument("--db", default=os.environ.get("SQLITE_PATH", str(REPO / "bench-data/gen318.db")))
    ap.add_argument("--sms-db", default=os.environ.get("SMS_DB_PATH", str(REPO / "bench-data/sms/sms.db")))
    ap.add_argument("--sms-config", default=os.environ.get("SMS_CONFIG_PATH", str(REPO / "config/sms.yaml")))
    ap.add_argument("--adapters", default=os.environ.get("ADAPTER_CONFIG_PATH", str(REPO / "config/adapters.yaml")))
    ap.add_argument("--device-id", default=os.environ.get("DEVICE_ID", "edge-node-001"))
    ap.add_argument("--check", metavar="FILE", help="check a record's checksum and stop")
    a = ap.parse_args()

    if a.check:
        ok = verify(json.loads(Path(a.check).read_text()))
        print("unchanged since it was made" if ok else "CHANGED: the checksum does not match")
        sys.exit(0 if ok else 1)
    if not a.month or not re.fullmatch(r"\d{4}-\d{2}", a.month):
        ap.error("give the month as YYYY-MM")
    if not Path(a.db).exists():
        sys.exit(f"no readings database at {a.db} (set --db or SQLITE_PATH)")

    import yaml

    from common.database import EdgeDatabase
    from sms.store import SmsStore

    cfg = {}
    if Path(a.sms_config).exists():
        cfg = yaml.safe_load(Path(a.sms_config).read_text()) or {}
    thresholds = cfg.get("thresholds") or {}
    db = EdgeDatabase(a.db, os.environ.get("SQLITE_ENCRYPTION_KEY", ""))
    db.connect()
    store = SmsStore(a.sms_db) if Path(a.sms_db).exists() else None
    record = build_record(
        db, store, a.month,
        site_name=str(cfg.get("site_name") or a.device_id), device_id=a.device_id,
        utc_offset_hours=float(cfg.get("utc_offset_hours", 1.0)),
        outage_voltage_v=float(thresholds.get("outage_voltage_v", 150.0)),
        register_map_verification=map_verification(a.adapters),
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", record["site_name"].lower()).strip("-") or "site"
    base = out / f"operating-record-{slug}-{a.month}"
    base.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    base.with_suffix(".csv").write_text(daily_csv(record))
    t = record["totals"]
    print(f"{record['site_name']}, {a.month}{'' if record['month_complete'] else ' (month not finished)'}")
    print(f"  the box saw {t['data_coverage_pct']}% of the month ({t['days_with_data']} days with data)")
    print(f"  power available {t['power_available_pct']}% of observed time, {t['outage_hours']} h off")
    print(f"  energy delivered {t['energy_delivered_kwh']} kWh, solar {t['solar_kwh']} kWh, genset {t['genset_kwh']} kWh")
    print(f"  alerts: {t['alerts']['critical']} critical, {t['alerts']['warning']} warning; labelled {t['alerts_labelled']}")
    print(f"  register maps verified to: {record['register_map_verification']}")
    print(f"wrote {base.with_suffix('.json')} and {base.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
