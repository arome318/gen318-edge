# Bench setup: Gen318 Edge on a real inverter

The goal of a bench test is to prove one vendor map against one real unit,
see live readings and trigger a safe test alert, before anything goes to a
site.

## What you need

| Item | Notes |
|---|---|
| A computer | Raspberry Pi 5 (4 GB+) with an SSD, any 64-bit Linux laptop, or a Mac for a first try |
| USB-RS485 adapter | For Deye, Sunsynk, Solis, SRNE, Growatt-style RS-485 ports. Isolated adapters are safer on site |
| Cable | Twisted pair from the inverter's RS-485/BMS-485 port (check the manual for A/B pins) to the adapter |
| Optional: GSM/LTE modem + SIM | SIM800 (2G), SIM7600 or Quectel EC25. Without it, use the `dry_run` SMS backend |

For Victron systems no adapter is needed: connect over Ethernet to the GX
device and enable Modbus TCP or MQTT on it.

## Option A: Linux or Raspberry Pi (Docker)

```bash
cp config/adapters.yaml.example config/adapters.yaml
cp config/sms.yaml.example config/sms.yaml
# edit both, then pass the RS-485 adapter into the collector service in
# docker-compose.yml (devices: - /dev/ttyUSB0:/dev/ttyUSB0)
docker compose up -d
docker compose logs -f collector sms
```

## Option B: Mac (no Docker)

Docker Desktop on macOS cannot pass USB serial devices into containers, so
the services run natively. A Mac with only USB-C ports needs a USB-C serial
adapter, or a USB-A adapter plus a USB-C to USB-A dongle.

```bash
tools/setup_mac.sh felicity     # installs tools, finds the adapter, writes configs
tools/run_bench.sh              # starts the box; page at http://localhost:8080
.venv/bin/python tools/check_readings.py   # compare with the inverter screen
```

On a Mac use the `/dev/cu.usbserial-...` port (not `/dev/tty.`). For a
Voltronic/Axpert-type inverter run `tools/setup_mac.sh voltronic`; for Deye
use the map name for your model, e.g. `tools/setup_mac.sh deye_sg04lp3`.

### Felicity (IVEM series)

- Use the RJ45 socket labelled **RS232**, next to the BMS socket. Do not
  unplug the BMS cable.
- It needs a USB-to-**RS232** adapter (FTDI or CH340), not RS485.
- Baud rate: 9600 first; if nothing reads, set `baudrate: 2400`.
- Solar is read per tracker (`solar.mppt1_power_w`); the map has no daily
  yield register and does not read battery current (it reads battery power).

### Try the whole thing without an inverter

```bash
brew install socat
socat pty,raw,echo=0,link=/tmp/inv pty,raw,echo=0,link=/tmp/box &
.venv/bin/python tools/felicity_emulator.py /tmp/inv &
# set port: /tmp/box in config/adapters.yaml, then
tools/run_bench.sh
```

## Example: one Deye hybrid inverter

```yaml
adapters:
  - name: deye
    enabled: true
    type: generic_modbus
    config:
      vendor_map: deye_sg04lp3      # pick the map for your model
      protocol: rtu
      port: /dev/ttyUSB0            # Mac: /dev/tty.usbserial-XXXX
      baudrate: 9600
      unit_id: 1
      asset_mappings:
        inverter: 7f3c2a10-0000-4000-8000-000000000001   # asset ids must be UUIDs
        battery: 7f3c2a10-0000-4000-8000-000000000001
        solar_array: 7f3c2a10-0000-4000-8000-000000000001
        grid_connection: 7f3c2a10-0000-4000-8000-000000000001
```

## Checks, in order

1. **Readings arrive.** Open `http://localhost:8080`. Battery SoC, PV power
   and output voltage should match the inverter's own screen.
2. **Values are right.** Compare at least five values with the inverter
   screen and note them. Wrong scale or sign means the map needs fixing.
3. **Status reply works.** With `dry_run`:
   `python tools/sms_inject.py bench-data/sms +2348030000009 POWER`, then
   read `bench-data/sms/outbox.jsonl`. With a modem, text POWER to the SIM.
4. **A safe test alert.** Unplug the RS-485 cable for 10 minutes: you should
   get "Box cannot read equipment". Plug it back: "OK again". Do not create
   real electrical faults to test alerts.
5. **Label it.** Reply `1` to the alert SMS (or inject `1` with
   `sms_inject.py`) and check the label on the local page.

## Record the result

Add a short validation note to the vendor map's YAML: date, inverter model
and firmware, which values matched, and who checked. That moves the map
from `community` to `bench`.
