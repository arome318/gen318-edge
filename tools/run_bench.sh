#!/usr/bin/env bash
# Run Gen318 Edge natively (no Docker) for bench tests, e.g. on a Mac whose
# USB-RS485 adapter Docker Desktop cannot pass through.
#
#   brew install mosquitto            # or: sudo apt install mosquitto
#   python3.12 -m venv .venv && .venv/bin/pip install -r edge/requirements-dev.txt
#   cp config/adapters.yaml.example config/adapters.yaml   # then edit it
#   cp config/sms.yaml.example config/sms.yaml             # then edit it
#   tools/run_bench.sh
#
# Then open http://localhost:8080 and watch bench-data/sms/outbox.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p bench-data/sms bench-data/models
# Every path below can be set in the environment first (tools/demo_video.sh
# does, to keep its files in a temporary folder).
export PYTHONPATH="$PWD/edge" MQTT_HOST=127.0.0.1 MQTT_PORT=1883 DEVICE_ID=${DEVICE_ID:-bench-001}
export SQLITE_PATH="${SQLITE_PATH:-$PWD/bench-data/gen318.db}" SQLITE_ENCRYPTION_KEY="" CLOUD_API_URL=""
export ADAPTER_CONFIG_PATH="${ADAPTER_CONFIG_PATH:-$PWD/config/adapters.yaml}"
export MODEL_PATH="${MODEL_PATH:-$PWD/bench-data/models}"
export SMS_CONFIG_PATH="${SMS_CONFIG_PATH:-$PWD/config/sms.yaml}"
export SMS_DB_PATH="${SMS_DB_PATH:-$PWD/bench-data/sms/sms.db}"
export STORAGE_RESERVE_BYTES=${STORAGE_RESERVE_BYTES:-104857600}

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if ! nc -z 127.0.0.1 1883 2>/dev/null; then
  mosquitto -p 1883 & pids+=($!)
  sleep 1
fi
# COLLECTOR_ENTRY lets tools/demo_video.sh start its own wrapper around the collector.
$PY ${COLLECTOR_ENTRY:--m collector.main} & pids+=($!)
$PY -m sms.main & pids+=($!)
$PY -m inference.main & pids+=($!)
$PY -m uvicorn api.main:app --host 0.0.0.0 --port 8080 & pids+=($!)
echo "Gen318 Edge bench running. Dashboard: http://localhost:8080  (Ctrl-C to stop)"
wait
