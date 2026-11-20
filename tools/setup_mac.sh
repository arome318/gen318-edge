#!/usr/bin/env bash
# One-time bench setup on a Mac, then writes configs for one inverter.
#
#   tools/setup_mac.sh felicity        # or: deye_sg04lp3, sunsynk_3p, voltronic, ...
#
# Needs Homebrew (https://brew.sh). Plug the USB serial adapter in first.
set -euo pipefail
cd "$(dirname "$0")/.."
MAP=${1:-felicity_hybrid}
[ "$MAP" = felicity ] && MAP=felicity_hybrid

command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
command -v mosquitto >/dev/null || brew install mosquitto
command -v python3.12 >/dev/null || brew install python@3.12
[ -d .venv ] || python3.12 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r edge/requirements-dev.txt

echo
echo "Serial adapters found:"
PORTS=$(ls /dev/cu.usbserial* /dev/cu.usbmodem* /dev/cu.wchusbserial* 2>/dev/null || true)
if [ -z "$PORTS" ]; then
  echo "  none. Plug in the USB-RS232/RS485 adapter (install its driver if it is CH340) and run again."
  exit 1
fi
echo "$PORTS" | nl
PORT=$(echo "$PORTS" | head -1)
echo "Using $PORT (edit config/adapters.yaml to change)"

mkdir -p config bench-data/sms
ASSET=$(.venv/bin/python -c "import uuid; print(uuid.uuid4())")
SITE=$(.venv/bin/python -c "import uuid; print(uuid.uuid4())")

if [ "$MAP" = voltronic ]; then
cat > config/adapters.yaml <<YAML
site_id: "$SITE"
tenant_id: "$SITE"
adapters:
  - name: inverter
    enabled: true
    type: voltronic_inverter
    config:
      port: $PORT
      baudrate: 2400
      asset_mappings:
        inverter: "$ASSET"
        battery: "$ASSET"
        solar_array: "$ASSET"
        grid_connection: "$ASSET"
YAML
else
cat > config/adapters.yaml <<YAML
site_id: "$SITE"
tenant_id: "$SITE"
adapters:
  - name: inverter
    enabled: true
    type: generic_modbus
    config:
      vendor_map: $MAP
      protocol: rtu
      port: $PORT
      # Felicity: if nothing reads at 9600, change to 2400 and restart.
      baudrate: 9600
      asset_mappings:              # one id for the whole system keeps alerts simple
        inverter: "$ASSET"
        battery: "$ASSET"
        solar_array: "$ASSET"
        grid_connection: "$ASSET"
        load: "$ASSET"
YAML
fi

if [ ! -f config/sms.yaml ]; then
  read -r -p "Your phone number for alerts (e.g. 0803...): " PHONE
  cat > config/sms.yaml <<YAML
site_name: Bench
default_lang: en
technicians:
  - {name: Me, phone: "$PHONE", lang: en}
quiet_hours: null
daily_summary_hour: 19
backend:
  type: dry_run                 # alerts go to bench-data/sms/outbox.jsonl
  directory: bench-data/sms
YAML
fi
echo
echo "Ready. Start the box with: tools/run_bench.sh"
echo "Then compare readings with the inverter screen: .venv/bin/python tools/check_readings.py"
