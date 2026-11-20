"""Kill the box at random and check that nothing is lost.

    python tools/torture_test.py --minutes 60          # the Docker stack
    .venv/bin/python tools/torture_test.py --writes 300   # the state files only

Mini-grid boxes lose power without warning. This test stands in for that on
a development machine:

* ``--minutes``: starts the real containers (plus a pretend inverter, see
  docker-compose.torture.yml) with 16 days of simulated history so the
  on-site models train, then for the given time kills one container's
  process at random (SIGKILL from outside it, no warning) or all of them in
  the same instant ("power cut"). Nothing is restarted by hand. It waits
  for the box to come back by itself, and checks that no stored reading,
  SMS record or learned model was lost or damaged.
* ``--writes``: starts a process that rewrites the learned-model files as
  fast as it can and kills it at a random moment, again and again, checking
  each time that the files on disk still load.

What it does not show: a SIGKILL stops a process, not the operating system.
A real power cut can also lose data the kernel had not yet written to the
card. Only a real box with its supply pulled can test that.

It runs as its own compose project (``gen318torture``) and removes that
project's volumes at the end. It does not touch a real box's data.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "bench-data" / "torture"
COMPOSE = ["docker", "compose", "-p", "gen318torture", "-f", "docker-compose.yml", "-f", "docker-compose.torture.yml"]
BOX = ["gen318-mqtt", "gen318-collector", "gen318-inference", "gen318-sync", "gen318-sms", "gen318-api",
       "gen318-watchdog"]
ALL = BOX + ["gen318-emulator"]
SEEDED_ASSET = "seeded-site"
LIVE_ASSET = "7f3c2a10-0000-4000-8000-0000000000aa"

SEED = f"""
import time
from common.database import EdgeDatabase
from inference.sitesim import SimSite
data = SimSite(days=16, seed=3).run()
shift = time.time() - max(ts[-1] for ts, _ in data.values())
db = EdgeDatabase('/data/gen318.db', ''); db.connect(); db.init_schema()
rows = []
for metric, (ts, vals) in data.items():
    rows += [(int((t + shift) * 1000), '{SEEDED_ASSET}', metric, float(v), 0) for t, v in zip(ts, vals)]
n = db.insert_measurements(rows); db.close(); print('SEEDED', n)
"""

PROBE = """
import glob, json, os, sqlite3
out = {}
try:
    c = sqlite3.connect('file:/data/gen318.db?mode=ro', uri=True, timeout=10)
    out['rows'], out['max_ts'] = c.execute('select count(*), max(timestamp) from measurements').fetchone()
    out['live_rows'] = c.execute("select count(*) from measurements where asset_id != 'seeded-site'").fetchone()[0]
    out['db_ok'] = c.execute('pragma integrity_check').fetchone()[0]
    c.close()
except Exception as e:
    out['db_ok'] = 'ERROR ' + repr(e)
try:
    s = sqlite3.connect('file:/sms/sms.db?mode=ro', uri=True, timeout=10)
    for t in ('alerts', 'deliveries', 'inbound', 'labels'):
        out[t] = s.execute(f'select count(*) from {t}').fetchone()[0]
    out['replies'] = s.execute("select count(*) from deliveries where kind='reply'").fetchone()[0]
    out['status_in'] = s.execute("select count(*) from inbound where kind='status'").fetchone()[0]
    out['sms_ok'] = s.execute('pragma integrity_check').fetchone()[0]
    s.close()
except Exception as e:
    out['sms_ok'] = 'ERROR ' + repr(e)
p = '/models/site_learning/meta.json'
try:
    meta = json.load(open(p))
    out['meta_ok'] = True
    out['model_kinds'] = sorted({v.get('kind') for v in meta.values() if isinstance(v, dict) and v.get('kind')})
except FileNotFoundError:
    out['meta_ok'], out['model_kinds'] = None, []
except Exception as e:
    out['meta_ok'], out['model_kinds'] = False, []
out['model_files'] = len(glob.glob('/models/site_learning/*.ubj'))
out['tmp_files'] = len(glob.glob('/models/site_learning/*.tmp'))
try:
    json.load(open('/metrics/energy.json')); out['energy_ok'] = True
except FileNotFoundError:
    out['energy_ok'] = None
except Exception:
    out['energy_ok'] = False
print(json.dumps(out))
"""


def sh(cmd: list[str], check: bool = True, timeout: int = 900) -> str:
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:6])}... failed: {r.stderr.strip()[-400:]}")
    return r.stdout.strip()


def running(names: list[str]) -> dict[str, bool]:
    out = sh(["docker", "inspect", "-f", "{{.Name}} {{.State.Running}}", *names], check=False)
    state = {n: False for n in names}
    for line in out.splitlines():
        name, _, flag = line.strip().lstrip("/").partition(" ")
        state[name] = flag == "true"
    return state


def kill_processes(names: list[str]) -> None:
    """SIGKILL each container's main process from outside it, all in one go."""
    pids = sh(["docker", "inspect", "-f", "{{.State.Pid}}", *names], check=False).split()
    pids = [p for p in pids if p.isdigit() and p != "0"]
    if pids:
        sh(["docker", "exec", "gen318-killer", "sh", "-c", "kill -9 " + " ".join(pids)], check=False)


def page_ok() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:8080/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def probe() -> dict:
    out = sh(["docker", "exec", "gen318-api", "python", "-c", PROBE], check=False, timeout=60)
    try:
        return json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        return {"probe_failed": True}


def restarts(names: list[str]) -> dict[str, int]:
    out = sh(["docker", "inspect", "-f", "{{.Name}} {{.RestartCount}}", *names], check=False)
    counts = {}
    for line in out.splitlines():
        name, _, n = line.strip().lstrip("/").partition(" ")
        counts[name] = int(n) if n.isdigit() else 0
    return counts


def wait_up(limit: float, killed: dict[str, int] | None = None) -> tuple[bool, float, list[str]]:
    """Wait until every container runs and the page answers. Returns (ok, seconds, still down).

    ``killed`` holds each victim's restart count from before the kill: a
    victim only counts as back once Docker has restarted it, so a process
    that has not been reaped yet is not mistaken for one that recovered.
    """
    t0 = time.time()
    down: list[str] = []
    while time.time() - t0 < limit:
        down = [n for n, up in running(ALL).items() if not up]
        if killed and not down:
            now = restarts(list(killed))
            down = [n for n, before in killed.items() if now.get(n, 0) <= before]
        if not down and page_ok():
            return True, time.time() - t0, []
        time.sleep(1)
    return False, time.time() - t0, down or ["page not answering"]


def inject_power(n: int) -> None:
    """A villager texts POWER (dry-run backend: a line in the inbox file)."""
    line = json.dumps({"from": f"+23480300{n:05d}", "text": "POWER"})
    sh(["docker", "exec", "gen318-sms", "sh", "-c", f"echo '{line}' >> /data/sms/inbox.jsonl"], check=False)


def write_configs() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "adapters.yaml").write_text(f"""site_id: "{LIVE_ASSET}"
tenant_id: "{LIVE_ASSET}"
adapters:
  - name: inverter
    enabled: true
    type: generic_modbus
    config:
      vendor_map: felicity_hybrid
      protocol: tcp
      host: emulator
      port: 5020
      asset_mappings:
        inverter: "{LIVE_ASSET}"
        battery: "{LIVE_ASSET}"
        solar_array: "{LIVE_ASSET}"
        grid_connection: "{LIVE_ASSET}"
        load: "{LIVE_ASSET}"
""")
    (WORK / "sms.yaml").write_text("""site_name: Torture test
default_lang: en
technicians:
  - {name: Technician, phone: "+2348030000001", lang: en}
quiet_hours: null
daily_summary_hour: null
hold_s: 10
status_per_hour: 3
thresholds:
  comms_lost_s: 90
  fresh_s: 120
backend:
  type: dry_run
  directory: /data/sms
""")


def docker_run(minutes: float, seed: int, keep: bool) -> int:
    rng = random.Random(seed)
    write_configs()
    log_path = WORK / "events.jsonl"
    log = log_path.open("w")
    events: list[dict] = []

    def note(**kw) -> None:
        kw["t"] = round(time.time() - started, 1)
        events.append(kw)
        log.write(json.dumps(kw) + "\n")
        log.flush()
        print(json.dumps(kw), flush=True)

    started = time.time()
    print("building and starting the stack...", flush=True)
    sh(COMPOSE + ["down", "-v", "--remove-orphans"], check=False)
    sh(COMPOSE + ["build", "-q"], timeout=1800)
    seeded = sh(COMPOSE + ["run", "--rm", "--no-deps", "-T", "inference", "python", "-c", SEED], timeout=600)
    print("seeded history:", next((x for x in seeded.splitlines() if x.startswith("SEEDED")), "?"), flush=True)
    sh(COMPOSE + ["up", "-d"], timeout=600)
    ok, secs, down = wait_up(240)
    if not ok:
        print("stack did not start:", down)
        return 2
    # wait for the on-site models to train from the seeded history
    t0 = time.time()
    base = probe()
    while time.time() - t0 < 300 and len(base.get("model_kinds", [])) < 4:
        time.sleep(10)
        base = probe()
    while time.time() - t0 < 420 and not base.get("live_rows"):
        time.sleep(10)
        base = probe()
    note(event="baseline", **base)
    if len(base.get("model_kinds", [])) < 4 or not base.get("live_rows"):
        print("baseline incomplete (models or live readings missing)")

    failures: list[str] = []
    injected = 0
    last = base
    end = time.time() + minutes * 60
    n = 0
    while time.time() < end:
        time.sleep(rng.uniform(45, 100))
        n += 1
        before = probe()
        if before.get("probe_failed"):
            before = last
        if rng.random() < 0.6:           # a villager texts just before the lights go
            injected += 1
            inject_power(injected)
            time.sleep(rng.uniform(0, 6))
        whole_box = rng.random() < 0.25
        victims = ALL if whole_box else [rng.choice(ALL)]
        counts = restarts(victims)
        kill_processes(victims)
        killed_at = time.time()      # after the kill: a row stamped later was written by the restarted box
        # Nothing is started by hand: the box has to come back by itself.
        ok, secs, down = wait_up(200, counts)
        helped = False
        if not ok:
            # did not come back by itself: note it, then start it so the test can go on
            helped = True
            sh(["docker", "start", *[d for d in down if d.startswith("gen318-")]], check=False)
            ok2, _, down2 = wait_up(120)
            failures.append(f"event {n}: {','.join(victims)} did not recover alone within 200 s ({down})")
            if not ok2:
                failures.append(f"event {n}: still down after a manual start: {down2}")
        # readings must resume: wait for a row stamped after the kill
        resumed, t0, after = False, time.time(), probe()
        while time.time() - t0 < 150:
            after = probe()
            if not after.get("probe_failed") and (after.get("max_ts") or 0) > killed_at * 1000:
                resumed = True
                break
            time.sleep(5)
        problems = []
        if after.get("probe_failed"):
            problems.append("could not read state")
        else:
            if after.get("db_ok") != "ok":
                problems.append(f"readings database: {after.get('db_ok')}")
            if after.get("sms_ok") != "ok":
                problems.append(f"SMS database: {after.get('sms_ok')}")
            for key in ("rows", "alerts", "deliveries", "inbound", "labels"):
                if (after.get(key) or 0) < (before.get(key) or 0):
                    problems.append(f"{key} went down {before.get(key)} -> {after.get(key)}")
            if after.get("meta_ok") is False:
                problems.append("learned-model index unreadable")
            if after.get("energy_ok") is False:
                problems.append("energy ledger unreadable")
            if not resumed:
                problems.append("no new reading within 150 s")
        failures += [f"event {n}: {p}" for p in problems]
        note(event="power cut (all)" if whole_box else "kill", victims=victims if not whole_box else ["all"],
             recovered_alone=not helped, seconds_to_recover=round(secs, 1),
             readings_resumed_s=round(time.time() - killed_at, 1) if resumed else None,
             rows=after.get("rows"), model_kinds=after.get("model_kinds"), meta_ok=after.get("meta_ok"),
             problems=problems)
        last = after if not after.get("probe_failed") else last

    time.sleep(40)   # let the last SMS replies go out
    final = probe()
    kills = [e for e in events if e["event"] != "baseline"]
    summary = {
        "minutes": minutes, "seed": seed, "events": len(kills),
        "single_kills": sum(1 for e in kills if e["event"] == "kill"),
        "whole_box_cuts": sum(1 for e in kills if e["event"] != "kill"),
        "recovered_alone": sum(1 for e in kills if e["recovered_alone"]),
        "median_seconds_to_recover": sorted(e["seconds_to_recover"] for e in kills)[len(kills) // 2] if kills else None,
        "max_seconds_to_recover": max((e["seconds_to_recover"] for e in kills), default=None),
        "by_victim": {},
        "rows_start": base.get("rows"), "rows_end": final.get("rows"),
        "live_rows_start": base.get("live_rows"), "live_rows_end": final.get("live_rows"),
        "db_ok": final.get("db_ok"), "sms_ok": final.get("sms_ok"),
        "model_kinds_start": base.get("model_kinds"), "model_kinds_end": final.get("model_kinds"),
        "model_files_start": base.get("model_files"), "model_files_end": final.get("model_files"),
        "leftover_tmp_files": final.get("tmp_files"),
        "power_texts_sent": injected, "power_texts_recorded": final.get("status_in"),
        "power_replies": final.get("replies"), "alerts": final.get("alerts"),
        "failures": failures,
    }
    for e in kills:
        for v in e["victims"]:
            d = summary["by_victim"].setdefault(v, {"kills": 0, "recovered_alone": 0, "worst_s": 0})
            d["kills"] += 1
            d["recovered_alone"] += int(e["recovered_alone"])
            d["worst_s"] = max(d["worst_s"], e["seconds_to_recover"])
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY\n" + json.dumps(summary, indent=2), flush=True)
    log.close()
    if not keep:
        sh(COMPOSE + ["down", "-v", "--remove-orphans"], check=False)
    return 1 if failures else 0


# ── state files under a hail of kills ───────────────────────────

WRITER = """
import sys, time
sys.path.insert(0, sys.argv[2])
import numpy as np
from common.energy import EnergyLedger, PowerModel
from inference.sitelearn import SiteLearner, fit_quantile
class NoHistory:
    def get_latest_readings(self): return []
    def get_readings_history(self, *a, **k): return []
sl = SiteLearner(NoHistory(), sys.argv[1])
ledger = EnergyLedger(sys.argv[1] + '/energy.json', PowerModel())
rng = np.random.default_rng()
X = rng.random((400, 2)); y = rng.random(400)
model = fit_quantile(X, y, 0.5, rounds=20)
print('ready', flush=True)
i = 0
while True:
    i += 1
    sl._store(f'pv:asset{i % 5}', model, {'trained_at': time.time(), 'kind': 'solar', 'i': i, 'pad': 'x' * 4000})
    ledger.add(0.3, 60)
"""


def writes_run(rounds: int, seed: int) -> int:
    import xgboost as xgb

    rng = random.Random(seed)
    work = tempfile.mkdtemp(prefix="gen318-writes-")
    site = Path(work) / "site_learning"
    bad: list[str] = []
    stores = 0
    for r in range(rounds):
        p = subprocess.Popen([sys.executable, "-c", WRITER, work, str(REPO / "edge")],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        p.stdout.readline()                       # "ready": it is now writing as fast as it can
        time.sleep(rng.uniform(0.002, 0.25))
        os.kill(p.pid, signal.SIGKILL)
        p.wait()
        try:
            meta = json.loads((site / "meta.json").read_text())
            stores = max(stores, max((v.get("i", 0) for v in meta.values()), default=0))
        except FileNotFoundError:
            meta = {}
        except ValueError as e:
            bad.append(f"round {r}: meta.json unreadable ({e})")
            continue
        for key in meta:
            f = site / (key.replace(":", "__") + ".ubj")
            try:
                xgb.Booster().load_model(str(f))
            except Exception as e:  # noqa: BLE001
                bad.append(f"round {r}: {f.name} does not load ({str(e)[:80]})")
        try:
            json.loads((Path(work) / "energy.json").read_text())
        except FileNotFoundError:
            pass
        except ValueError:
            bad.append(f"round {r}: energy.json unreadable")
    result = {"rounds": rounds, "kills": rounds, "damaged_files": len(bad), "details": bad[:10],
              "writes_completed_in_a_single_round_at_most": stores}
    print(json.dumps(result, indent=2))
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "writes.json").write_text(json.dumps(result, indent=2))
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--minutes", type=float, help="kill containers at random for this long")
    ap.add_argument("--writes", type=int, help="kill a state-file writer this many times")
    ap.add_argument("--seed", type=int, default=318)
    ap.add_argument("--keep", action="store_true", help="leave the stack running afterwards")
    a = ap.parse_args()
    if a.writes:
        sys.exit(writes_run(a.writes, a.seed))
    if a.minutes:
        sys.exit(docker_run(a.minutes, a.seed, a.keep))
    ap.error("give --minutes or --writes")


if __name__ == "__main__":
    main()
