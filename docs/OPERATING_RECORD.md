# Monthly operating record

A file the operator can hand to a lender, guarantor or programme: what the
site did in one month, made on the box from its own readings. Nothing is
sent anywhere. The operator decides who gets it.

```bash
python tools/operating_record.py 2026-11 --out records/
python tools/operating_record.py --check records/operating-record-kuje-2026-11.json
```

It writes a `.json` file with everything and a `.csv` with one row per day.
Make each month's record within a couple of weeks of the month ending: a
standalone box keeps 45 days of readings (`LOCAL_RETENTION_DAYS`).

## What is in it

| Field | Meaning |
|---|---|
| `schema` | `gen318-edge/operating-record/1` |
| `site_name`, `device_id`, `month`, `utc_offset_hours` | Which site and which month, in the site's local time |
| `generated_at`, `month_complete` | When it was made, and whether the month had ended |
| `register_map_verification` | `community`, `bench` or `field`: how far the register maps the readings came through have been checked on real equipment. The weakest map counts. |
| `totals.data_coverage_pct` | Share of the month the box actually has readings for |
| `totals.power_available_pct` | Share of observed time the inverter output was at or above `outage_voltage_v` |
| `totals.outage_hours` | Observed time below it |
| `totals.energy_delivered_kwh`, `solar_kwh`, `genset_kwh` | Energy, from power readings |
| `totals.lowest_battery_pct` | Lowest state of charge seen |
| `totals.alerts`, `alerts_labelled`, `sms_sent` | Alerts raised, what technicians said about them, SMS sent by kind |
| `daily[]` | The same figures for each day |
| `alerts[]` | Each alert: code, severity, raised, cleared, minutes open, technician label |
| `record_sha256` | Checksum of the record |

## Read it with these in mind

- **Unknown is not zero.** Time is counted in 15-minute slots. A slot with
  no reading is unknown: it is not counted as power on, and not as power
  off. A day the box was off shows empty cells, not 0% or 100%. Read every
  figure next to `data_coverage_pct`.
- **Energy is measured at the inverter**, not at customer meters. It is
  delivered energy, not billed energy.
- **The figures are only as good as the register map.** Until a map is
  marked `bench` or `field`, treat the numbers as unverified.
- **The checksum is not a signature.** It shows the file was not changed
  after it was made. It does not prove who made it, and someone who can
  edit the box can make a new record with a new checksum.
- **No phone numbers or technician notes** are in the record.

## What it is not

It is not the format of any particular funder or verification platform.
Nigeria's DARES programme verifies claims through the Odyssey platform, and
this record has not been checked against what Odyssey accepts. It is a
plain, documented format that can be mapped to one.
