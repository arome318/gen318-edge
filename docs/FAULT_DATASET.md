# Open fault dataset: data sheet

Every box keeps a record of the alerts it raised and what the technician
said about each one (1 real, 2 false alarm, 3 other cause). Exported and
pooled across sites, that is a labelled dataset of faults on real African
mini-grids, which does not exist in the open today.

```bash
python tools/export_fault_dataset.py --out dataset/
```

This writes `faults.jsonl`, `faults.csv` and `summary.json` on the box and
sends nothing anywhere. **Status, November 2026: the tool and the format
are ready; there are no field labels yet.** The first will come from the
pilot.

## One row

| Field | Meaning |
|---|---|
| `site` | A made-up id, the same for every export from one box. It cannot be turned back into the box or the site. |
| `code` | What the box flagged, e.g. `battery_low`, `pv_underperform`, `comms_lost` |
| `severity` | `critical` or `warning` |
| `month`, `weekday`, `hour` | When, in the site's local time: `2026-11`, Monday = 0, hour of day. No date. |
| `minutes_open` | How long until it cleared (empty if still open) |
| `label` | `real`, `false_alarm`, `other_cause`, or empty if no technician answered |
| `minutes_to_label` | How long the technician took to answer |
| `values` | The numbers in the alert, e.g. state of charge |
| `before` | The hour before the alert, averaged: battery state of charge, voltage, power and temperature; solar and load power; inverter temperature; output voltage; genset power |

Schema id: `gen318-edge/fault-dataset/1`.

## What is left out, and why

- **Phone numbers:** the box never stores them.
- **Technician notes:** free text can contain names.
- **Site name, device id, asset ids, alert references:** they point at a place.
- **Exact dates and times:** with a date, an outage can be matched to a
  village. Month, weekday and hour keep what a model needs.

This lowers the risk of identifying a site; it does not remove it. A site
with an unusual pattern in a small dataset may still be recognisable to
someone who knows it. Each operator should read their export before sharing.

## Who decides

The operator of each site. The export is a file on their box. Gen318 does
not receive it unless they send it.

## What it is good for

- Measuring how often each kind of alert is a false alarm, per site and overall.
- Training or checking models that tell real faults from false alarms.
- Setting starting thresholds for new sites from what worked elsewhere.

## What it is not good for

- **Finding missed faults.** There is a row only when the box raised an
  alert. A fault the box never flagged is not in the data, so the dataset
  cannot give a detection rate.
- **Ground truth.** A label is one technician's judgement by SMS, sometimes
  given before anyone reached the site. "False alarm" can mean "I could not
  find it".
- **Comparing sites directly.** Thresholds differ per site and move with
  the technician's replies (see docs/SITE_LEARNING.md), so the same code
  does not mean the same condition everywhere.
- **Anything about unanswered alerts.** Technicians answer some alerts and
  not others, and not at random.

## Licence

Not yet chosen. A permissive data licence (CC BY 4.0) is the intention, to
be agreed with the first operators who contribute.
