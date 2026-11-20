"""Export this box's alerts and technician labels as rows for an open fault dataset.

    python tools/export_fault_dataset.py --out dataset/

Writes faults.jsonl, faults.csv and summary.json, and sends nothing
anywhere. The rows carry no phone numbers, names, site name, device id or
exact dates (see edge/sms/dataset.py for what is left out and why, and
docs/FAULT_DATASET.md for what the dataset is good for). Read the files
before you share them: you decide whether and where they are published.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "edge"))

from sms.dataset import build_rows, summary, to_csv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Export alerts and technician labels for an open fault dataset.")
    ap.add_argument("--out", default="fault-dataset", help="folder to write into")
    ap.add_argument("--db", default=os.environ.get("SQLITE_PATH", str(REPO / "bench-data/gen318.db")))
    ap.add_argument("--sms-db", default=os.environ.get("SMS_DB_PATH", str(REPO / "bench-data/sms/sms.db")))
    ap.add_argument("--sms-config", default=os.environ.get("SMS_CONFIG_PATH", str(REPO / "config/sms.yaml")))
    ap.add_argument("--device-id", default=os.environ.get("DEVICE_ID", "edge-node-001"))
    ap.add_argument("--labelled-only", action="store_true", help="leave out alerts no technician answered")
    a = ap.parse_args()
    if not Path(a.sms_db).exists():
        sys.exit(f"no SMS record at {a.sms_db} (set --sms-db or SMS_DB_PATH)")

    import yaml

    from common.database import EdgeDatabase
    from sms.store import SmsStore

    cfg = yaml.safe_load(Path(a.sms_config).read_text()) or {} if Path(a.sms_config).exists() else {}
    db = None
    if Path(a.db).exists():
        db = EdgeDatabase(a.db, os.environ.get("SQLITE_ENCRYPTION_KEY", ""))
        db.connect()
    rows = build_rows(SmsStore(a.sms_db), db, a.device_id, float(cfg.get("utc_offset_hours", 1.0)),
                      labelled_only=a.labelled_only)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "faults.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    (out / "faults.csv").write_text(to_csv(rows))
    info = summary(rows)
    (out / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"{info['alerts']} alerts, {info['labelled']} labelled {info['labels']}")
    for code, d in sorted(info["by_code"].items()):
        print(f"  {code}: {d['alerts']} alerts, {d['real']} real, {d['false_alarm']} false alarm, {d['other_cause']} other")
    if not info["labelled"]:
        print("No technician labels yet: the rows are there, the labels are empty.")
    print(f"wrote {out}/faults.jsonl, faults.csv and summary.json. Read them before sharing.")


if __name__ == "__main__":
    main()
