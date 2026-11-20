# Gen318 Edge

**Open, offline fault detection for solar mini-grids and hybrid power sites.**

Gen318 Edge is a small box that sits at a power site. It reads the
inverters, batteries, meters and gensets on site, spots faults on the box
itself, and sends plain-language SMS alerts to the technician. People on the
mini-grid can text **POWER** to the box to find out if their power is on and,
if not, why. It needs no internet and no cloud account.

It was built in Nigeria, where many mini-grids run on 2G, technicians cover
many sites, and a battery fault noticed late means a dark village and a
diesel bill.

## What it does

- **Reads mixed equipment.** Victron (Modbus TCP and MQTT), SunSpec inverters
  (Fronius, SMA, SolarEdge), and Modbus register maps for Deye, Sunsynk,
  Solis, SRNE, ATESS, GoodWe, Sungrow, Felicity and genset controllers
  (SmartGen, DSE, ComAp), plus Voltronic/Axpert and CAN batteries.
- **Keeps the data on site.** Readings go to a local SQLite database
  (encrypted if you set a key) and survive 30+ days without a connection.
- **Detects faults locally.** Rules for low and critical battery, battery and
  inverter overheating, cell imbalance, inverter error codes, overload,
  power off, genset running when the battery is full, and lost
  communication.
- **Learns each site on the box.** After 7 to 14 days, three small XGBoost
  models train on the site's own data: the solar panels' best output by
  time of day, the battery's normal voltage under load, and the normal load
  band. They flag dirty or shaded panels, a battery sagging under load
  (ageing cells or loose terminals) and unusual load (illegal connections,
  tripped feeders). No data leaves the box. See `docs/SITE_LEARNING.md`,
  including results on simulated sites and their limits.
- **Sends SMS through a modem on the box.** Works on 2G. Alerts in English and
  Nigerian Pidgin (Hausa, Yoruba and Igbo planned). Warnings wait out quiet
  hours; critical alerts never wait. A daily summary at 7 pm.
- **Answers anyone who texts POWER**, with the site status, the cause of an
  outage and whether the technician knows. Rate-limited to save airtime.
- **Learns from technicians.** A technician replies `1` (real), `2` (false
  alarm) or `3` (other cause) to an alert. Labels are stored on the box and
  can be exported as an anonymised fault dataset. Phone numbers are only
  ever stored as salted hashes.
- **Shows a local page** at `http://<box>:8080` on the site Wi-Fi: open
  alerts, labels and the latest readings.

## What it never does

- It never switches, starts or stops equipment. It is advisory only: the
  remote-control code from the original Gen318 node was removed from this
  release.
- It never sends data anywhere unless you set `CLOUD_API_URL` to a server
  you choose. Left empty (the default), nothing leaves the box except the
  SMS you configure.

## Try it in one minute (no hardware)

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r edge/requirements-dev.txt
.venv/bin/python tools/demo_site.py
```

This plays a simulated evening at a mini-grid (battery falls, power cuts,
technician confirms, power returns) and prints every SMS that would be sent.

## Run it on a site

1. Hardware: a Raspberry Pi 5 (4 GB or more) or any 64-bit Linux box, a
   USB-RS485 adapter for RS-485 inverters, and a GSM/LTE USB modem or HAT
   with a SIM for SMS. See `docs/BENCH_SETUP.md`.
2. `cp config/adapters.yaml.example config/adapters.yaml` and describe your
   equipment (see `docs/ONBOARDING_DEYE_HYBRID.md` for a worked Deye example).
3. `cp config/sms.yaml.example config/sms.yaml` and add your technicians,
   community contacts and modem port.
4. `docker compose up -d`, then open `http://<box>:8080`.

## Status

This is early. The software is tested (920+ automated tests, simulators and
emulated field protocols), but **every vendor map is marked `community`:
transcribed from public sources and not yet polled on a physical unit.**
Each map moves to `bench` and then `field` only with a validation record.
The first field pilot is under way in Nigeria.

## Layout

| Path | What |
|---|---|
| `edge/collector/` | Equipment adapters and vendor register maps |
| `edge/inference/` | On-site learning (`sitelearn.py`), site simulator, model runtime |
| `edge/sms/` | Fault rules, SMS alerts, POWER replies, technician labels |
| `edge/api/` | Local page and JSON API on port 8080 |
| `edge/sync/` | Optional store-and-forward sync to your own server |
| `edge/watchdog/` | Container health, tamper switch |
| `tools/` | Demo, bench runner, SMS injector, energy measurement |

## Tests

```bash
cd edge && ../.venv/bin/python -m pytest -q
```

## Contributing

The most useful contributions right now:

- **Verify a vendor map** on a real inverter you have access to, and send
  the result (see `edge/collector/adapters/vendor_maps/README.md`).
- **Fix or add SMS templates** in your language (`edge/sms/templates/`).
  The Pidgin file is a draft waiting for technician review.
- **Report false alarms** and the thresholds that worked on your site.

## Sustainability

How the project is paid for, what stays free forever, and what happens if
Gen318 changes course: see `SUSTAINABILITY.md`.

## Licence

Apache License 2.0. See `LICENSE` and `NOTICE`.
