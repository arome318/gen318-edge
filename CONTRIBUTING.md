# Contributing to Gen318 Edge

Thank you for helping. You do not need to be a programmer: a technician who
checks one register on one inverter, or corrects one SMS in their language,
moves this project further than most code changes.

## Ways to help, most useful first

1. **Verify a vendor map on real equipment.** Every map ships as
   `community` (transcribed from public sources, never polled on a physical
   unit). Run the box against your inverter, compare a few values with the
   inverter's own screen, and send what you found. The steps and the record
   format are in `edge/collector/adapters/vendor_maps/README.md`.
2. **Correct or add SMS wording** in `edge/sms/templates/`. One YAML file
   per language. Keep each message ASCII and under 160 characters. The
   Pidgin file is a draft; Hausa, Yoruba and Igbo are wanted.
3. **Report a false alarm or a missed fault**, with the site's thresholds
   and what the equipment was doing. Open an issue; leave out phone numbers
   and anything that identifies a customer.
4. **Code.** See below.

## Setting up

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r edge/requirements-dev.txt
cd edge && ../.venv/bin/python -m pytest -q
```

This works on Linux and macOS; no hardware, Docker or SIM is needed. `tools/demo_site.py` plays a simulated evening and prints the SMS.

## Rules for changes

These are the promises in `README.md` and `SUSTAINABILITY.md`, as they
apply to a pull request:

- **Advisory only.** Nothing may write to an inverter, battery, genset or
  breaker. Adapters read. A change that adds a write will not be merged.
- **Nothing leaves the box by default.** No new network call may run unless
  the operator has set `CLOUD_API_URL` or configured the SMS modem.
- **No phone numbers in clear.** Store the salted hash (`sms/store.py`).
- **Say what was tested on what.** If a number comes from the simulator,
  say so next to the number. Do not describe a `community` map as supported.
- **Tests come with the change.** A fix has a test that fails without it.
- **Small and plain.** The box runs on a Raspberry Pi and is read by people
  for whom English is a second language. Prefer the simple version.

Changes to the on-site models should come with the output of
`python tools/sitelearn_eval.py --seeds 10` before and after.

## Licence

Contributions are accepted under Apache-2.0, the licence of the project,
with no copyright assignment.
