# On-site learning

Each Gen318 Edge box learns what normal looks like at its own site, from its
own readings, on the box. Nothing is downloaded and nothing leaves the site.

## The three models

| Model | Learns | Trains after | Flags | Method |
|---|---|---|---|---|
| Solar | Best-case output for each time of day | 7 days | Panels below 75% of their own learned best for 5 days (dust, shade, dead string) | XGBoost quantile regression (90th percentile) on time of day |
| Battery | Battery voltage given state of charge, power and temperature | 14 days (then frozen) | Extra voltage drop that grows with load (ageing cells, loose or corroded terminals) | XGBoost regression baseline; the residual is regressed on load, with state of charge as a covariate |
| Load | Normal band of site load by time of day and weekday | 7 days (retrained daily) | Load far above the band for most of 2 hours (illegal connection, feeder fault) or far below it with power on (tripped feeder) | XGBoost quantile regression (10th, 50th, 90th percentiles) |

Each model is small, trains in seconds on a Raspberry Pi and is stored in
`/models/site_learning/` with a `meta.json` describing what it learned. The
battery baseline is frozen so slow ageing is not absorbed into "normal";
reset it after replacing the battery. The solar model retrains monthly,
leaving out days it had already flagged.

Findings are warnings only. They reach technicians by SMS like any other
alert, wait out quiet hours, and can be labelled by reply (1 real, 2 false
alarm, 3 other cause). Those labels are how thresholds get tuned per site.

## How well it works (simulated sites only)

Results from `python tools/sitelearn_eval.py --seeds 10`: ten simulated
60-day sites per scenario, checked every 6 hours (hourly at night for the
load cases). 16 November 2026.

| Scenario | Detected | Median days to detect | False alarms |
|---|---|---|---|
| Clean site (530 site-days) | n/a | n/a | 0 |
| Panels lose 40% (heavy dust) | 10/10 | 2.75 | 0 |
| Panels lose 20% | 3/10 | 10.75 | 0 |
| Battery resistance doubles | 10/10 | 4 | 0 |
| Battery resistance rises 50% | 10/10 | 5 | 0 |
| 800 W extra load at night | 10/10 | same night (about 2 hours) | 0 |
| 400 W extra load at night | 0/10 | not detected | 0 |

**Read these with care.** The simulator is a simplified physical model
(clear-sky solar with day-to-day cloud, a 48 V LFP bank with internal
resistance, an evening-peak village load). It shows the method behaves as
designed. It does not show field accuracy: real sites have messier data,
gaps, seasons (Harmattan haze lowers output for weeks) and equipment quirks.
Small faults (20% dust, 400 W) are mostly missed by design, to keep false
alarms down. Field validation starts with the first pilot.
