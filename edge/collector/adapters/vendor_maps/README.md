# Vendor register maps

Declarative Modbus register maps for `generic_modbus`, one YAML file per
device family. The adapter type string stays `generic_modbus` — adding a
brand here needs no registry change and no new code.

A site selects one with:

```yaml
adapters:
  - name: deye-hybrid
    type: generic_modbus
    config:
      vendor_map: deye_sg04lp3
      protocol: rtu
      port: /dev/ttyUSB0
      asset_mappings:
        inverter: <asset uuid>
        battery: <asset uuid>
        solar_array: <asset uuid>
        grid_connection: <asset uuid>
```

## Rules every file in this directory obeys

- **`verification: community | bench | field`** is mandatory. Everything
  shipped today is `community`: transcribed from open-source projects and
  exercised against a simulator, never polled on a physical unit. A map at
  `community` **may not be described as "supported" in any material**
  (vendor integrations spec §10). Promotion needs a validation record, not
  an edit to this field alone.
- **Every register line names its source** in a `# src:` comment — the
  project, file and pinned revision it was transcribed from. A value the
  source does not settle carries the exact marker
  `# UNVERIFIED — bench required`.
- **A map may only use metric names in the backend's `ALL_METRICS`.** The
  edge never imports the backend, so that is enforced by a backend test,
  not at runtime here.
- **No write declarations.** The loader refuses any `write_mappings`,
  `writes`, `write_registers` or `write` key. Vendor maps are read-only by
  construction; the legacy installer-declared write path on `generic_modbus`
  is separate and is not reachable from a map.
- **`extends: <name>`** is the only inheritance mechanism (ordinary YAML
  anchors work inside one file but never across files). Depth is bounded
  and cycles are refused.
- A file whose addresses are blank is a **template** (the ComAp case): it
  loads for inspection and refuses to poll until a technician fills the
  addresses from that controller's own exported table.

`Dockerfile.arm64` copies `collector/` wholesale, so files added here ship
in the collector image with no build change.
