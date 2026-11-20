"""Pretend a phone texted the box (dry_run backend only).

    python tools/sms_inject.py /path/to/sms-dir +2348030000009 POWER
"""

import json
import sys
from pathlib import Path

directory, phone, text = sys.argv[1], sys.argv[2], " ".join(sys.argv[3:])
with (Path(directory) / "inbox.jsonl").open("a") as fh:
    fh.write(json.dumps({"from": phone, "text": text}) + "\n")
print(f"queued {text!r} from {phone}")
