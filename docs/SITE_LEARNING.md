# On-site learning

Each Gen318 Edge box learns what normal looks like at its own site, from its
own readings, on the box. Nothing is downloaded and nothing leaves the site.

## The models

| Model | Learns | Trains after | Flags | Method |
|---|---|---|---|---|
| Solar | Best-case output for each time of day | 7 days | Panels below 75% of their own learned best for 5 days (dust, shade, dead string) | XGBoost quantile regression (90th percentile) on time of day |
| Battery | Battery voltage given state of charge, power and temperature | 14 days (then frozen) | Extra voltage drop that grows with load (ageing cells, loose or corroded terminals) | XGBoost regression baseline; the residual is regressed on load, with state of charge as a covariate |
| Load | Normal band of site load by time of day and weekday | 7 days (retrained daily) | Load far above the band for most of 2 hours (illegal connection, feeder fault) or far below it with power on (tripped feeder) | XGBoost quantile regression (10th, 50th, 90th percentiles) |
| Run-out | The battery's usual net discharge by time of day and weekday, and how many watt-hours one percent of charge is worth | 7 days (retrained daily) | The bank is likely to reach its cut-off within 6 hours | XGBoost quantile regression (median), walked forward from the present state of charge |

Each model is small, trains in seconds on a Raspberry Pi and is stored in
`/models/site_learning/` with a `meta.json` describing what it learned. The
battery baseline is frozen so slow ageing is not absorbed into "normal";
reset it after replacing the battery. The solar model retrains monthly,
leaving out days it had already flagged.

Findings are warnings only. They reach technicians by SMS like any other
alert, wait out quiet hours (except the run-out warning, which is no use in
the morning), and can be labelled by reply (1 real, 2 false alarm, 3 other
cause).

## A full battery is not a dirty panel

When the bank is full, an off-grid inverter throttles the panels down to
the load. Low solar output then means there was nothing to charge, not that
the panels are dirty. The solar model therefore leaves out every 15-minute
slot where the state of charge is at or above 95% (`pv_full_soc_pct`), both
when it learns and when it judges. A day with too few free slots gives no
verdict, so a site that is full by mid-morning all week is simply not
judged that week; `meta.json` records the share of slots left out
(`curtailed_pct`). If the inverter reports no state of charge the model
behaves as it did before and can be fooled by a fall in daytime load.

## The run-out warning

One model per battery learns its usual net discharge (load minus solar) for
each time of day. The box also works out the bank's usable size from its
own readings: watt-hours discharged divided by percent of charge lost
(`usable_kwh_estimate` in `meta.json`). From the present state of charge it
walks the usual night forward. If the forecast passes the cut-off
(`SITE_RUNOUT_SOC_PCT`, default 10%) by at least 5 points within 6 hours,
the technician gets "Battery likely to run out around 02:15".

Two limits are deliberate. A night forecast to end only a point or two
below the cut-off is not reported, so shallow run-outs are missed. And a
site whose bank has emptied on 4 of the last 7 days gets no nightly
warning: that is a sizing problem the technician already knows about.

The forecast is the site's median night. It does not know that tonight is a
festival or that a feeder is off.

## How technician replies change sensitivity

Three or more false-alarm replies (2) among the last five for a warning,
with no "real" reply (1) among them, make that warning one step less
sensitive at that site. Five in a row make it two steps, which is the
most. Any "real" reply among the last five, or `RESET TUNING` from a
technician, puts it back. Technicians are told by SMS each time, and the
local page lists what has changed.

| Warning | One step |
|---|---|
| Battery low | 2.5 points lower, never within 2 points of the critical level |
| Cell imbalance | 20% wider spread |
| Inverter hot | 3 C higher |
| Overload | 3 points higher, never above 100% |
| Genset running on a charged battery | 5 points higher, never above 90% |
| Solar underperforming | 5 points lower ratio |
| Battery sagging under load | 25% larger drop |
| Load high / low | 0.15 higher factor / 0.05 lower factor |
| Battery run-out | warns 1.5 hours later (never less than 2 hours ahead) |

Critical alerts (power off, battery critical, battery hot, inverter fault,
lost readings) are never loosened. The step sizes are first guesses, not
field-tuned values.

## Power cuts

These boxes lose power without warning. Model files and `meta.json` are
written to a temporary file and renamed, so a cut leaves either the old
file or the new one. A file that is damaged anyway is dropped and learned
again from the readings on the box. For the battery baseline that means a
new baseline from the last 16 days, which can hide ageing that happened
before the loss.

## How well it works (simulated sites only)

Results from `python tools/sitelearn_eval.py --seeds 10`: ten simulated
60-day sites per scenario, checked every 6 hours (hourly at night for the
load cases, every 30 minutes for run-out). 20 November 2026. The simulated
inverter now throttles the panels when the bank is full, as a real one does.

| Scenario | Detected | Median days to detect | False alarms |
|---|---|---|---|
| Clean site (530 site-days) | n/a | n/a | 0 |
| Panels lose 40% (heavy dust) | 10/10 | 2.5 | 0 |
| Panels lose 20% | 2/10 | 20.5 | 0 |
| Battery resistance doubles | 10/10 | 4 | 0 |
| Battery resistance rises 50% | 10/10 | 5 | 0 |
| 800 W extra load at night | 10/10 | same night (about 2 hours) | 0 |
| 400 W extra load at night | 0/10 | not detected | 0 |
| Daytime load halves, panels healthy | n/a | n/a | 0 dirty-panel findings |

The last row is the case the curtailment rule exists for. Before it, the
same scenario raised a dirty-panel warning on every simulated site within
about four days.

Run-out warning, ten 60-day sites per bank size on the same load. A "power
cut" is the output going off because the bank is empty. "Wrong" counts
warnings not followed by a bank at or below 15% within 8 hours.

| Bank | Power cuts | Warned beforehand | Median hours of warning | Warnings sent | Wrong |
|---|---|---|---|---|---|
| 17 kWh | 20 | 10 | 7.0 | 10 | 0 |
| 18 kWh | 6 | 6 | 7.3 | 6 | 0 |
| 20 kWh | 2 | 2 | 6.8 | 2 | 0 |
| 24 kWh | 1 | 1 | 7.7 | 1 | 0 |

On the 17 kWh sites half the cuts came without a warning: those nights were
forecast to end too close to the cut-off to call. A 15 kWh bank on this
load empties nearly every night and, by design, gets no nightly warning.

**Read these with care.** The simulator is a simplified physical model
(clear-sky solar with day-to-day cloud, a 48 V LFP bank with internal
resistance, an evening-peak village load). It shows the method behaves as
designed. It does not show field accuracy: real sites have messier data,
gaps, seasons (Harmattan haze lowers output for weeks) and equipment quirks.
The simulated load is the same shape every night, which flatters the
run-out forecast. Small faults (20% dust, 400 W) are mostly missed by
design, to keep false alarms down; 20% dust is caught less often than
before the curtailment rule (2 of 10 sites, was 3 of 10), because slots
with a full bank no longer count. Field validation starts with the first
pilot.

## Experiment: thresholds set to a stated false-alarm rate (not in the box)

The unusual-load warning uses hand-set numbers. `tools/calibration_eval.py`
asks what happens if the box instead measures how far its own site strays
from the learned band on normal days, and sets the threshold where only a
chosen share of normal 2-hour windows would cross it (split conformal
calibration). Nothing in the box uses this yet. Ten simulated sites, days
30 to 60, 20 November 2026:

| Threshold | Normal windows over it | False alarms per 30 days | 400 W night load caught | 800 W caught |
|---|---|---|---|---|
| Set for 1% of windows | 1.07% | 9.3 | 10/10, in 1.0 h | 10/10, in 1.0 h |
| Set for 0.1% of windows | 0.18% | 1.8 | 10/10, in 1.2 h | 10/10, in 1.2 h |
| Today's fixed rule | 0.00% | 0 | 0/10 | 10/10, in 1.2 h |

What it shows: the method hits the rate it is asked for, and it catches
the 400 W load the fixed rule misses by design. What it costs: windows
overlap, so even "0.1% of windows" is about two false SMS a month, where
the fixed rule sent none. That trade is the operator's to make, which is
the point of exposing it as a number.

Why it is not in the box: the simulator's load noise is the same kind
every day, which is the condition under which the method's promise holds.
A real village has market days, festivals and seasons. It needs a field
trial, alongside the fixed rule, before it sends anyone an SMS.

