#!/usr/bin/env bash
# A filmable demo of Gen318 Edge on a Mac: no Docker, no inverter, no SIM.
#
#   tools/demo_video.sh             start everything and open the local page
#   tools/demo_video.sh unplug      (second terminal) pull the inverter cable
#   tools/demo_video.sh plug        (second terminal) put the cable back
#   tools/demo_video.sh text POWER  (second terminal) a villager texts the box
#
# What runs: a pretend Felicity inverter (tools/felicity_emulator.py) on a
# virtual serial cable made by socat, and the real box software reading it
# (tools/run_bench.sh). SMS are not sent anywhere: the dry_run backend writes
# them to a file, and the local page shows them. Everything the demo writes
# goes to a temporary folder that is deleted when you press Ctrl-C.
#
# The inverter values are made up. This shows the software working end to
# end; it does not show that the Felicity register map matches a real unit.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
PY="$REPO/.venv/bin/python"
POINTER="${TMPDIR:-/tmp}/gen318-demo-current"
TECH_PHONE="+2348030000001"       # made-up numbers, as in config/sms.yaml.example
VILLAGER_PHONE="+2348030000009"
URL="http://localhost:8080"

say()  { printf '%s\n' "$*"; }
fail() { printf 'demo_video: %s\n' "$*" >&2; exit 1; }

demo_dir() {
  [ -f "$POINTER" ] || fail "the demo is not running. Start it first: tools/demo_video.sh"
  local dir
  dir=$(cat "$POINTER")
  [ -d "$dir" ] || fail "the demo folder is gone. Start the demo again: tools/demo_video.sh"
  printf '%s' "$dir"
}

start_emulator() {  # $1 = demo folder
  "$PY" "$REPO/tools/felicity_emulator.py" "$1/inv" >> "$1/emulator.log" 2>&1 &
  echo $! > "$1/emulator.pid"
}

stop_emulator() {   # $1 = demo folder
  if [ -f "$1/emulator.pid" ]; then
    kill "$(cat "$1/emulator.pid")" 2>/dev/null || true
    rm -f "$1/emulator.pid"
  fi
}

case "${1:-start}" in
  unplug)
    DEMO=$(demo_dir)
    [ -f "$DEMO/emulator.pid" ] || fail "the inverter is already unplugged."
    stop_emulator "$DEMO"
    say "Inverter unplugged. In about 35 seconds the box texts the technician:"
    say "  \"Box cannot read equipment ...\"   Watch the SMS table on $URL"
    exit 0 ;;
  plug)
    DEMO=$(demo_dir)
    [ -f "$DEMO/emulator.pid" ] && fail "the inverter is already plugged in."
    start_emulator "$DEMO"
    say "Inverter plugged back in. Within about 20 seconds the box texts: \"OK again ...\""
    exit 0 ;;
  text)
    DEMO=$(demo_dir)
    shift
    [ $# -ge 1 ] || fail "say what to text, for example: tools/demo_video.sh text POWER"
    "$PY" "$REPO/tools/sms_inject.py" "$DEMO/sms" "$VILLAGER_PHONE" "$@" > /dev/null
    say "A villager texted \"$*\" to the box. The reply shows on $URL in a few seconds."
    exit 0 ;;
  tech)
    DEMO=$(demo_dir)
    shift
    [ $# -ge 1 ] || fail "say what the technician replies, for example: tools/demo_video.sh tech 1"
    "$PY" "$REPO/tools/sms_inject.py" "$DEMO/sms" "$TECH_PHONE" "$@" > /dev/null
    say "The technician replied \"$*\". The box's answer shows on $URL in a few seconds."
    exit 0 ;;
  start) ;;
  *) fail "unknown command '$1'. Use: (nothing), unplug, plug, text <words>, tech <words>" ;;
esac

# ── 1. tools ───────────────────────────────────────────────────
[ "$(uname)" = Darwin ] || fail "this script is written for macOS. On Linux use tools/run_bench.sh."
for tool in socat mosquitto; do
  # mosquitto is installed under sbin, which is not always on PATH
  if ! command -v "$tool" > /dev/null && ! [ -x "$(brew --prefix 2>/dev/null)/sbin/$tool" ]; then
    command -v brew > /dev/null || fail "Homebrew is needed to install $tool. Get it from https://brew.sh"
    say "Installing $tool with Homebrew..."
    brew install "$tool"
  fi
done
export PATH="$(brew --prefix 2>/dev/null)/sbin:$(brew --prefix 2>/dev/null)/bin:$PATH"

if [ ! -x "$PY" ]; then
  say "Setting up Python (once, takes a few minutes)..."
  BASE_PY=$(command -v python3.12 || command -v python3) || fail "Python 3.12 is needed: brew install python@3.12"
  "$BASE_PY" -m venv "$REPO/.venv"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r "$REPO/edge/requirements-dev.txt"
fi

# ── 2. nothing else on the ports ───────────────────────────────
if [ -f "$POINTER" ] && [ -d "$(cat "$POINTER")" ]; then
  fail "the demo is already running in another terminal. Press Ctrl-C there first."
fi
if nc -z 127.0.0.1 8080 2> /dev/null; then
  fail "port 8080 is in use, so the local page cannot start. Stop what is using it (for the Docker stack: docker compose down) and run this again."
fi

# ── 3. temporary files ─────────────────────────────────────────
DEMO=$(mktemp -d "${TMPDIR:-/tmp}/gen318-demo.XXXXXX")
mkdir -p "$DEMO/sms" "$DEMO/models"
echo "$DEMO" > "$POINTER"
ASSET=$("$PY" -c "import uuid; print(uuid.uuid4())")

cat > "$DEMO/adapters.yaml" <<YAML
site_id: "$ASSET"
tenant_id: "$ASSET"
adapters:
  - name: inverter
    enabled: true
    type: generic_modbus
    config:
      vendor_map: felicity_hybrid
      protocol: rtu
      port: $DEMO/box
      baudrate: 9600
      asset_mappings:
        inverter: "$ASSET"
        battery: "$ASSET"
        solar_array: "$ASSET"
        grid_connection: "$ASSET"
        load: "$ASSET"
YAML

# The Felicity map polls every 30 seconds, and a site config cannot change
# that. For the camera, the demo reads a copy of the maps with 5 seconds
# instead (see tools/demo_collector.py). Registers and scales are untouched.
cp -R "$REPO/edge/collector/adapters/vendor_maps" "$DEMO/vendor_maps"
sed -i '' 's/poll_interval_seconds: 30/poll_interval_seconds: 5/' "$DEMO/vendor_maps/felicity_hybrid.yaml"
grep -q 'poll_interval_seconds: 5' "$DEMO/vendor_maps/felicity_hybrid.yaml" \
  || fail "could not shorten the poll interval in the demo's copy of the Felicity map."

cat > "$DEMO/sms.yaml" <<YAML
site_name: Demo Mini-grid
default_lang: en
technicians:
  - {name: Technician, phone: "$TECH_PHONE", lang: en}
community:
  - {name: Village contact, phone: "+2348030000002", lang: pcm}
quiet_hours: null
daily_summary_hour: null
hold_s: 5                 # real sites use 120
thresholds:
  comms_lost_s: 30        # real sites use 600 (10 minutes)
  fresh_s: 60
backend:
  type: dry_run           # nothing is sent: SMS go to a file and the local page
  directory: $DEMO/sms
YAML

# ── 4. start: cable, inverter, box ─────────────────────────────
BOX_PID=""
SOCAT_PID=""
cleanup() {
  trap - EXIT INT TERM
  [ -n "$BOX_PID" ] && kill "$BOX_PID" 2> /dev/null || true
  stop_emulator "$DEMO"
  [ -n "$SOCAT_PID" ] && kill "$SOCAT_PID" 2> /dev/null || true
  wait 2> /dev/null || true
  rm -f "$POINTER"
  case "$DEMO" in */gen318-demo.*) rm -rf "$DEMO" ;; esac
  say ""
  say "Demo stopped. Temporary files removed."
}
trap cleanup EXIT INT TERM

socat "pty,raw,echo=0,link=$DEMO/inv" "pty,raw,echo=0,link=$DEMO/box" 2> "$DEMO/socat.log" &
SOCAT_PID=$!
for _ in $(seq 1 50); do [ -e "$DEMO/inv" ] && [ -e "$DEMO/box" ] && break; sleep 0.1; done
[ -e "$DEMO/box" ] || fail "socat could not make the virtual serial cable (see $DEMO/socat.log)."
start_emulator "$DEMO"

SQLITE_PATH="$DEMO/gen318.db" MODEL_PATH="$DEMO/models" \
ADAPTER_CONFIG_PATH="$DEMO/adapters.yaml" SMS_CONFIG_PATH="$DEMO/sms.yaml" \
SMS_DB_PATH="$DEMO/sms/sms.db" SMS_TICK_SECONDS=3 PAGE_REFRESH_SECONDS=3 \
DEVICE_ID="demo-box" PY="$PY" \
DEMO_VENDOR_MAPS_DIR="$DEMO/vendor_maps" COLLECTOR_ENTRY="$REPO/tools/demo_collector.py" \
  "$REPO/tools/run_bench.sh" > "$DEMO/box.log" 2>&1 &
BOX_PID=$!

say "Starting the box..."
for _ in $(seq 1 60); do
  curl -s -o /dev/null "$URL/health" && break
  kill -0 "$BOX_PID" 2> /dev/null || { tail -20 "$DEMO/box.log" >&2; fail "the box stopped while starting (log above)."; }
  sleep 1
done
curl -s -o /dev/null "$URL/health" || fail "the local page did not come up in 60 seconds (see $DEMO/box.log)."
[ -n "${NO_OPEN:-}" ] || open "$URL"

cat <<TEXT

==================================================================
 Gen318 Edge demo is running.   Local page: $URL
 The page refreshes every 3 seconds. SMS appear in its top table.
==================================================================

 Give it about 15 seconds for the first readings, then film.
 Readings on the page change every 5 seconds (a real site: 30).
 Run these in a SECOND terminal window, from this folder:

 1. "Unplug" the inverter (the comms-lost SMS appears in ~35 s):

      tools/demo_video.sh unplug

 2. A villager texts POWER to the box (reply appears in ~5 s):

      tools/demo_video.sh text POWER

 Extras, if you want them on camera:

      tools/demo_video.sh plug        cable back in: "OK again" SMS
      tools/demo_video.sh tech 1      technician marks the alert as real
      tools/demo_video.sh text WUTA   the same question, another word

 Nothing here is sent to a phone or to the internet.
 Press Ctrl-C in THIS window to stop and clean up.

TEXT
wait "$BOX_PID"
