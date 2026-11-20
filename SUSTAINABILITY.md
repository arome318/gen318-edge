# How Gen318 Edge stays alive

Gen318 Edge is free and open, and stays that way. This page explains who
pays for its upkeep, what is promised regardless of who pays, and what
happens if the company behind it changes course.

## What is free, forever

Everything in this repository: the box software, the inverter register
maps, the fault rules, the on-site learning models, the SMS service, the
simulator and the evaluation tools. Apache-2.0, no "open core" catch inside
it: a site running only this repository gets every alert and every model.

These promises do not depend on any funder or customer:

1. **No relicensing.** Code released here stays Apache-2.0. Contributions are
   accepted under the same licence, with no copyright assignment.
2. **Data stays with the site.** Nothing leaves a box unless its operator
   points it at a server of their choosing. No paid feature will require
   sending raw data to Gen318.
3. **Advisory only.** The box never switches equipment. That will not change
   to unlock a paid feature.
4. **A way out.** If Gen318 stops maintaining the project, the repository,
   register maps and published datasets move to a neutral home (below) so
   anyone can carry on.

## Who pays for the upkeep

Upkeep is about one engineer's time: reviewing contributed register maps,
fixing bugs, keeping dependencies current and answering installers. It is
paid for in three ways, in order of how soon they arrive.

| Source | What it pays for | Status (Nov 2026) |
|---|---|---|
| **Grants for open energy-access tools** | Building the first versions and validating them in the field | Applying (Mozilla Human-Scale AI, Nov 2026). Other energy-access and open-source funds to follow |
| **Gen318's paid services, built on top** | Ongoing maintenance, from day one of the first paid site | Price list set (below); no paying customers yet |
| **Funders who need the records** | Boxes and monitoring across the sites they finance | Offer drafted for lenders, guarantors and results-based programmes; not yet sold |

### Gen318's paid services

Operators who want more than one box gives them pay Gen318 for a fleet
service on top of the open edge: a dashboard across many sites, work orders,
and the monthly operating records that DARES-funded mini-grids and
commercial solar clients must give their funders and regulators. Gen318's
list price, in naira and fixed for 12 months:

| Site size | Per site, per month |
|---|---|
| Under 50 kW (fleets of 3+ sites) | ₦45,000 |
| 50–250 kW | ₦70,000 |
| 250 kW – 1 MW | ₦125,000 (proposed, not yet quoted) |

That is a small share of a mini-grid's running cost: for a 100 kW site,
₦70,000 a month is roughly 4% of total operating cost at AMDA's published
benchmark of $137–155 per kW per year. The open edge alone stays free; the
paid service is optional and uses the same box.

### Funders who need the records

Lenders, guarantors and results-based programmes in Nigeria have promised
their own funders operating results (connections, energy delivered,
uptime) from sites they finance. They can pay for boxes and monitoring
across their portfolio while each operator keeps its data and decides what
the funder sees. This puts the cost on the party that most needs the
evidence, not on operators with thin margins.

### Hardware

Operators or their funders buy their own boxes: a bench unit is a
Raspberry Pi 5 and USB adapters; a field unit with isolated serial ports, a
clock battery and a modem costs about $370. Grant money goes to people and
evidence, not equipment.

## Governance

- **Today:** maintained by Gen318 (Abraham Arome Onoja). Decisions and
  roadmap are made in public issues.
- **Technician advisory group:** NEMSA-certified technicians review alert
  wording, fault guides and thresholds before they ship.
- **Within 12 months:** move the project under a fiscal sponsor or neutral
  open-source home for energy projects (candidates: an energy-access
  open-source foundation such as EnAccess, or LF Energy), separate from the
  company, with at least one maintainer outside Gen318.
- **Compatibility over lock-in:** follow open standards where they exist
  (SunSpec today; open metering and gateway efforts such as OpenAMI as they
  mature).

## What could still go wrong

Being honest about the risks is part of the plan:

- No paying customers yet. Until the first ones arrive, upkeep depends on
  grants.
- Small operators have thin margins; that is why funders, not operators, are
  the main payer we are designing for.
- Rugged hardware is expensive. Cheaper options (an old Android phone as the
  gateway) are on the roadmap.
