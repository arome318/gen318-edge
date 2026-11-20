# Onboarding a Deye or SunSynk hybrid

For a technician standing in front of the inverter. Everything here is local
Modbus over the inverter's own RS485 port: no cloud account, no vendor
registration, no terms to accept.

**Read this first.** These maps are `verification: community`. They are
transcribed from open-source projects and exercised against a simulator. No
one has yet read one on a physical unit, so commissioning is not a formality
— three specific readings below are unconfirmed and you are the one who
confirms them. Until that is done, Deye is not "supported" and must not be
described that way to a customer.

That applies equally to the **single-phase** maps (`deye_sg03lp1`,
`deye_sg02lp1`), added later at the same community grade. They were
cross-checked against Deye's own protocol table as well as three community
projects, which is more than the three-phase maps had — but a table is not a
unit, and no single-phase inverter has been read yet. Section 4a is the
single-phase commissioning list.

## 1. Before you touch anything

| Check | Why |
|---|---|
| Which port? | Use the port marked **Modbus/RS485**. **Do not use the BMS port** — it carries the battery's own CAN/RS485 conversation. The DB9 logger port is RS-232 level and will not work. |
| Which pins? | **Read this unit's own manual.** Sources disagree: two give pin 7 A+ / pin 8 B−, another gives pin 1 B− / pin 2 A+ / pin 3 GND. Pinout varies by model and year. Write down which pins you used — it goes in the validation record. |
| Is a logger already on the bus? | A Solarman stick polling the same RS485 line will contend with the Pi. Either remove it or expect retries. Note which you did. |
| Slave id | Should be 1. Some units ship as 0; set it to 1 in the inverter menu. |
| Serial settings | 9600 8N1. |

## 2. Site configuration

A hybrid is ONE device and FOUR assets. That is not bureaucracy: ingest
rejects a battery metric addressed to an inverter asset, so the roles have to
be bound to real asset UUIDs before anything arrives.

```yaml
adapters:
  - name: deye-hybrid
    type: generic_modbus
    config:
      vendor_map: deye_sg04lp3      # or deye_sg01hp3 (HV), sunsynk_3p (SunSynk),
                                    # deye_sg03lp1 / deye_sg02lp1 (single-phase)
      protocol: rtu
      port: /dev/ttyUSB0
      baudrate: 9600
      unit_id: 1
      asset_mappings:
        inverter: <inverter asset uuid>
        battery: <battery asset uuid>
        solar_array: <solar array asset uuid>
        grid_connection: <grid connection asset uuid>
        # Optional. Bind only if the site actually has them:
        # load: <load/feeder asset uuid>
        # diesel_genset: <genset asset uuid>
      # features:
      #   gen_port: true            # ONLY if a genset is wired to the gen port
```

Which map:

- `deye_sg04lp3` — SUN-5/8/10/12K-SG04LP3, three-phase low voltage.
- `deye_sg01hp3` — SUN-30/50K-SG01HP3, three-phase high voltage. Different
  battery and PV scaling, four MPPTs.
- `sunsynk_3p` — SunSynk three-phase. Identical hardware to the LV Deye; the
  alias exists so identity and reporting say SunSynk.

Single-phase (LP1) units, community grade — pick by **rated power**, not by
the suffix (SG01LP1 exists at both 8 kW and 16 kW):

- `deye_sg03lp1` — up to 8 kW: SUN-3.6K/5K/6K-SG03LP1, SUN-5K/6K/8K-SG05LP1,
  SUN-5K/8K-SG01LP1. Two MPPTs.
- `deye_sg02lp1` — 10-16 kW: SUN-10K/12K-SG02LP1, SUN-14K/16K-SG01LP1. Three
  MPPTs, and three readings that stay **off** until you choose their scale
  (section 4a.5).

The single-phase register range (0-249) is different from the three-phase
one (500-700). Never put a three-phase map on a single-phase unit or the
reverse: it produces numbers, all of them wrong. The probe's identity block
tells you which you have (section 4a.1).

## 3. Prove it reads before you leave

```bash
python -m collector.register_probe --vendor-map deye_sg04lp3 \
    --protocol rtu --port /dev/ttyUSB0 --baudrate 9600 --unit-id 1
```

This reads every register the map declares, once, and writes a validation
record to the node's local database. It never writes to the inverter.

Reading the output:

| Column | What it means |
|---|---|
| `raw` | The words the inverter actually sent. This is the evidence; everything else is interpretation. |
| `value` | What the map made of those words. |
| `ok` | Read and decoded. |
| `no_value` | Read, but the map declined to publish — an unmapped enum code, or a value the vendor marks as not-implemented. The reason is printed. |
| `read_error` | The inverter did not answer that address. A handful is normal (not every model implements every register). All of them means a wrong slave id, wrong port or wrong wiring. |
| `OUTSIDE DECLARED BOUNDS` | Decoded, but implausible. Usually a scale problem. Do not ignore it. |

## 4. The three things only you can confirm

The probe cannot settle these. They need the site.

**4.1 Which way is import?** The map assumes a positive reading on register
625 means importing from the grid. Put a known load on the site with the grid
present, then compare `grid.import_power_w` against the site's own meter. If
the platform shows export while the meter shows import, the convention is
inverted for this firmware — record that and stop; do not edit the map on
site.

**4.2 Which way is charging?** Same test for `battery.power_w` (register
590). The platform's convention is **positive = charging into the bank**.
Charge the battery and check the sign.

**4.3 Site load or backup load?** Registers 650-653 are called plain "Load"
by one source and the UPS/backup load by another. Put a load on a
non-backup circuit and watch whether `load.power_w` moves. If it does not,
the register is measuring the backup output only — record that, and disable
those registers for this site:

```yaml
      registers_disable: [load_w, load_l1_w, load_l2_w, load_l3_w]
```

## 4a. Single-phase units: what only you can confirm

Run the probe with the single-phase map you expect:

```bash
python -m collector.register_probe --vendor-map deye_sg03lp1 \
    --protocol rtu --port /dev/ttyUSB0 --baudrate 9600 --unit-id 1
```

**4a.1 Is it the right map?** Look at the identity block. `device_type`
should read 3 (single-phase hybrid); 5 means a three-phase LV unit — stop and
use `deye_sg04lp3`. `rated_power_w` decides between the two single-phase
maps: up to 8000 is `deye_sg03lp1`, 10000-16000 is `deye_sg02lp1`. If the
rated power says 10-16 kW, re-run the probe with `--vendor-map deye_sg02lp1`
before going further.

**4a.2 Which way is import?** Register 169 is taken as positive = import,
because Deye's own table says so. With the grid present and a load running,
compare `grid.import_power_w` with the inverter's display and the site's own
meter. Then, if the site exports, check that `grid.export_power_w` moves
instead. If they are the wrong way round, record it and stop.

**4a.3 Which grid point is it?** 169 is the power at the **inverter's own
grid terminals**. If the house has circuits wired on the grid side of the
inverter (non-essential circuits) and an external CT is fitted on the main
supply, the site's real import is register 172, which the map does not read.
Compare `grid.import_power_w` with the main meter while a non-essential
circuit is running: if they differ by that circuit, write it down — the
platform's grid figure is then the inverter's grid port only.

**4a.4 Which way is charging?** The map **negates** battery power (190) and
current (191), because Deye reports them as battery *output* (positive =
discharging) and the platform's convention is positive = charging. Charge the
battery (from PV or grid) and check that `battery.power_w` and
`battery.current_a` are both positive. If they are negative while the display
says charging, record it and stop; do not edit the map on site.

**4a.5 10-16 kW units only: pick the scales.** On the larger units the
sources disagree whether grid power (169) and load power (178) count in 1 W
or 10 W, and whether battery current (191) counts in 0.01 A or 0.1 A. The map
ships those three readings **off**. The probe reads every candidate side by
side: `grid_w` / `grid_w_x10`, `load_w` / `load_w_x10`, `bat_i` / `bat_i_x10`.
Compare each pair with the inverter's display at a moment when the value is
large enough to tell a 10× difference (a few hundred watts or more), then
enable exactly one of each pair:

```yaml
      vendor_map: deye_sg02lp1
      features:
        power_scale_10w: true          # or power_scale_1w: true
        battery_current_0_01a: true    # or battery_current_0_1a: true
```

Enabling both of a pair is refused at start-up — the collector logs
`vendor_map_rejected` and reads nothing from that inverter — because the two
would collide on one measurement. One source scales grid power but not load
power on 16 kW units. If that is what you see (grid matches at 10 W, load at
1 W), enable both power features and disable the wrong register of each:

```yaml
      features:
        power_scale_1w: true
        power_scale_10w: true
      registers_disable: [grid_w, load_w_x10]
```

**4a.6 Load port.** Register 178 measures the inverter's LOAD (backup)
port. Circuits on the grid side of the inverter never pass through it, so
`load.power_w` is backed-up load, not whole-house consumption. Bind the
`load` role to an asset that represents the backed-up circuits, or leave it
unbound.

**4a.7 Lifetime grid import.** Compare `grid.import_energy_kwh` with the
display's lifetime "bought" (or "import") total. It is decoded from two
registers that are not next to each other (78 and 80); a value tens of
millions of kWh too high means the decode is wrong.

**4a.8 Generator port.** Enable `gen_port` only if a genset is wired to the
GEN port **and** the inverter's AUX/GEN port is set to generator input (not
smart load). Run the genset and check that `genset.power_w` is positive. A
negative reading is kept but marked suspect; record it.

## 5. What to write down

The validation record already holds the map revision, the identity the
inverter reported and every register's result. Add, by hand:

- your name and the date
- the **port and pins** you used
- model, serial and firmware from the inverter's own display
- the outcome of each check in section 4
- at least five values compared against the inverter's display

For a single-phase unit also record: the `device_type` and `rated_power_w`
the probe printed, which map you chose, and — on a 10-16 kW unit — which
scale features you enabled and the display readings you compared them with.

Then come back after a day and run the probe again: a single sweep cannot
show that a lifetime counter is increasing, and that is one of the bench
gate's requirements.

## 6. Batteries over CAN

If the pack talks CAN rather than presenting itself over the inverter's
Modbus (FreedomWon, Pylontech-class), there are two paths.

**Easier: the inverter's BMS mirror.** A Deye already republishes the
battery's SOC, voltage, current and temperature from that CAN conversation
into its own registers, and the map reads them. No extra hardware. This is
the right answer for almost every site.

**Direct tap: a CAN HAT on the Pi.** Only where the mirror is not enough.

> **The Pi must never transmit on that bus.** Two active masters on a BMS bus
> is how a battery disconnects itself. The interface has to be in listen-only
> mode *before* the collector starts, and the adapter refuses to open the bus
> unless it can confirm that:
>
> ```bash
> ip link set can0 type can bitrate 500000 listen-only on
> ip link set up can0
> ip -details link show can0   # must print: listen-only on
> ```
>
> The adapter checks and never fixes. It will not reconfigure an interface it
> shares with an inverter and a battery.

The CAN reader runs as its own opt-in container, `can-bridge`, not as an
adapter inside the collector. A SocketCAN interface belongs to a network
namespace, so only a container sharing the host's can see it — and this keeps
that to one small process instead of the collector, which holds every vendor
adapter's configuration.

Set the interface up on the host first, then start the profile:

```bash
# On the Pi, before starting the container:
ip link set can0 type can bitrate 500000 listen-only on
ip link set up can0
ip -details link show can0      # must print: listen-only on

# Then, in the compose directory:
EDGE_CAN_ASSET_ID=<battery asset uuid> docker compose --profile can up -d
```

Settings, all optional except the asset id:

| Variable | Default | Notes |
|---|---|---|
| `EDGE_CAN_ASSET_ID` | none | The battery asset. The bridge refuses to start without it |
| `EDGE_CAN_INTERFACE` | `can0` | |
| `EDGE_CAN_BITRATE` | `500000` | `250000` for Ingeteam profiles |
| `EDGE_CAN_PROFILE` | `victron_generic` | or `byd` |
| `EDGE_CAN_REQUIRE_LISTEN_ONLY` | `true` | Only ever set false on an isolated bench bus |

A node without a CAN tap never runs this: it is behind the `can` profile and
is not started by a plain `docker compose up`.

RJ45 CAN is pin 7 CAN-H, pin 8 CAN-L, with a 120 Ω terminator on the first
and last unit.

Note what this reports and what it does not: `battery.state` is **coarse** —
0 for nothing flagged, 1 for the pack flagging something. The two public
descriptions of that frame disagree about what each bit means, so the
platform reports only what both agree on. The raw bytes are kept with the
reading for whoever settles it later.

## 7. Known limits

- **HP3 units**: an upstream report from October 2025, still unanswered, says
  some HP3 inverters return wrong battery voltage and wrong PV values at the
  documented scaling. If the probe disagrees with the inverter's display,
  believe the display and record the difference.
- **No writes.** These maps cannot control the inverter, and the adapter
  refuses to start if a site config pairs a vendor map with write mappings.
- **No cloud.** Deye Cloud and Solarman OpenAPI are deliberately not
  integrated.
- **Single-phase gaps**: daily counters and lifetime load energy are read by
  every source but have no platform metric; the BMS charge limits (312-315)
  and lifetime generator energy are disputed and not read; the external-CT
  grid reading (172) is not read (section 4a.3).

Related: `docs/ONBOARDING_GENSET_CONTROLLER.md`,
`docs/vendor-integrations/SOURCES.md` (where every register came from),
`docs/vendor-integrations/VERIFICATION.md` (what is and is not validated).
