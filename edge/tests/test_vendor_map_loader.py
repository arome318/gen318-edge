"""Vendor register-map loader (vendor integrations spec §2.2, plan Task 2).

A vendor map is a declarative YAML file shipped inside the collector
container. The loader turns it into validated register specs the generic
Modbus engine polls. These tests pin the schema contract: override
precedence, bounded ``extends:`` inheritance, explicit community metadata,
role binding, disabled entries, template refusal, path safety and the
absence of any write declaration.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from collector.adapters import vendor_map as vm
from collector.adapters.vendor_map import (
    VendorMapError,
    load_vendor_map,
    resolve_vendor_map,
)

BATTERY = "10000000-0000-0000-0000-000000000001"
INVERTER = "10000000-0000-0000-0000-000000000002"
SOLAR = "10000000-0000-0000-0000-000000000003"
GRID = "10000000-0000-0000-0000-000000000004"
GENSET = "10000000-0000-0000-0000-000000000005"


BASE_MAP = textwrap.dedent(
    """
    vendor: testvendor
    model_family: base
    source: synthetic fixture (unit test)
    verification: community
    protocol_defaults:
      protocol: rtu
      baudrate: 9600
      parity: N
      stopbits: 1
      bytesize: 8
      unit_id: 1
      function_code: 3
      address_base: 0
    identity:
      serial: {address: 3, type: string, words: 5}
    assets: [inverter, battery]
    optional_assets: [diesel_genset]
    features:
      gen_port: false
    groups:
      - name: power
        poll_interval_seconds: 30
        registers:
          - {id: inv_state, metric: inverter.state, address: 500, type: enum,
             values: {0: 0, 1: 1, 2: 2}, asset: inverter}
          - {id: bat_v, metric: battery.voltage_v, address: 587, type: uint16,
             scale: 0.01, asset: battery, min: 0, max: 1000}
          - {id: gen_w, metric: genset.power_w, address: 667, type: int16,
             asset: diesel_genset, feature: gen_port}
      - name: energy
        poll_interval_seconds: 300
        registers:
          - {id: bat_chg, metric: battery.charge_energy_kwh, address: 516,
             type: uint32_le, scale: 0.1, asset: battery}
    """
)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(text)
    return path


@pytest.fixture
def maps_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write(tmp_path, "base_map", BASE_MAP)
    monkeypatch.setattr(vm, "VENDOR_MAPS_DIR", tmp_path)
    return tmp_path


def _bindings(**extra: str) -> dict[str, str]:
    return {"inverter": INVERTER, "battery": BATTERY, **extra}


# ── Loading and metadata ────────────────────────────────────────


def test_load_resolves_metadata_and_defaults(maps_dir: Path) -> None:
    resolved = resolve_vendor_map("base_map", {"asset_mappings": _bindings()})

    assert resolved.vendor == "testvendor"
    assert resolved.model_family == "base"
    assert resolved.verification == "community"
    assert resolved.protocol["protocol"] == "rtu"
    assert resolved.protocol["baudrate"] == 9600
    assert resolved.protocol["unit_id"] == 1
    assert resolved.required_roles == ("inverter", "battery")
    assert resolved.optional_roles == ("diesel_genset",)
    assert resolved.complete is True
    assert resolved.missing_addresses == ()
    assert [g.name for g in resolved.groups] == ["power", "energy"]
    assert [g.poll_interval_seconds for g in resolved.groups] == [30, 300]


def test_verification_is_required_and_bounded(maps_dir: Path) -> None:
    _write(maps_dir, "no_verif", BASE_MAP.replace("verification: community\n", ""))
    with pytest.raises(VendorMapError, match="verification"):
        load_vendor_map("no_verif")

    _write(
        maps_dir,
        "bad_verif",
        BASE_MAP.replace("verification: community", "verification: supported"),
    )
    with pytest.raises(VendorMapError, match="verification"):
        load_vendor_map("bad_verif")


def test_loader_rejects_path_traversal_and_unknown_names(maps_dir: Path) -> None:
    for bad in ("../base_map", "base_map/../x", "Base-Map", "", "sub/base"):
        with pytest.raises(VendorMapError, match="map name"):
            load_vendor_map(bad)
    with pytest.raises(VendorMapError, match="not found"):
        load_vendor_map("does_not_exist")


# ── Registers and roles ─────────────────────────────────────────


def test_registers_resolve_to_bound_assets(maps_dir: Path) -> None:
    resolved = resolve_vendor_map("base_map", {"asset_mappings": _bindings()})
    specs = {r.id: r for g in resolved.groups for r in g.registers}

    assert specs["bat_v"].asset_id == BATTERY
    assert specs["bat_v"].metric == "battery.voltage_v"
    assert specs["bat_v"].scale == pytest.approx(0.01)
    assert specs["bat_v"].min_value == 0
    assert specs["bat_v"].max_value == 1000
    assert specs["bat_chg"].data_type == "uint32_le"
    assert specs["bat_chg"].word_count == 2
    assert specs["inv_state"].enum_values == {0: 0.0, 1: 1.0, 2: 2.0}
    # Group defaults flow down to each register.
    assert specs["bat_v"].function_code == 3
    assert specs["bat_v"].unit_id == 1
    # The feature-gated genset register is absent while gen_port is off.
    assert "gen_w" not in specs


def test_unbound_required_role_refuses(maps_dir: Path) -> None:
    with pytest.raises(VendorMapError, match="battery"):
        resolve_vendor_map("base_map", {"asset_mappings": {"inverter": INVERTER}})


def test_role_binding_must_be_a_uuid(maps_dir: Path) -> None:
    with pytest.raises(VendorMapError, match="UUID"):
        resolve_vendor_map(
            "base_map", {"asset_mappings": _bindings(battery="not-a-uuid")}
        )


def test_unbound_optional_role_drops_its_registers(maps_dir: Path) -> None:
    resolved = resolve_vendor_map(
        "base_map", {"asset_mappings": _bindings(), "features": {"gen_port": True}}
    )
    ids = {r.id for g in resolved.groups for r in g.registers}
    assert "gen_w" not in ids
    assert "diesel_genset" in resolved.dropped_roles


def test_feature_flag_enables_gated_register(maps_dir: Path) -> None:
    resolved = resolve_vendor_map(
        "base_map",
        {
            "asset_mappings": _bindings(diesel_genset=GENSET),
            "features": {"gen_port": True},
        },
    )
    specs = {r.id: r for g in resolved.groups for r in g.registers}
    assert specs["gen_w"].asset_id == GENSET


def test_unknown_role_in_register_rejected(maps_dir: Path) -> None:
    _write(
        maps_dir,
        "bad_role",
        BASE_MAP.replace("asset: battery, min", "asset: fridge, min"),
    )
    with pytest.raises(VendorMapError, match="role"):
        load_vendor_map("bad_role")


# ── Site overrides ──────────────────────────────────────────────


def test_site_settings_override_protocol_defaults(maps_dir: Path) -> None:
    resolved = resolve_vendor_map(
        "base_map",
        {
            "asset_mappings": _bindings(),
            "protocol": "tcp",
            "host": "192.168.1.50",
            "port": 1502,
            "unit_id": 7,
            "baudrate": 19200,
        },
    )
    assert resolved.protocol["protocol"] == "tcp"
    assert resolved.protocol["host"] == "192.168.1.50"
    assert resolved.protocol["port"] == 1502
    assert resolved.protocol["unit_id"] == 7
    specs = {r.id: r for g in resolved.groups for r in g.registers}
    assert specs["bat_v"].unit_id == 7


def test_site_can_disable_and_add_registers(maps_dir: Path) -> None:
    resolved = resolve_vendor_map(
        "base_map",
        {
            "asset_mappings": _bindings(),
            "registers_disable": ["bat_chg"],
            "registers_add": [
                {
                    "id": "site_extra",
                    "metric": "battery.soc_pct",
                    "address": 588,
                    "type": "uint16",
                    "asset": "battery",
                    "group": "power",
                }
            ],
        },
    )
    ids = {r.id for g in resolved.groups for r in g.registers}
    assert "bat_chg" not in ids
    assert "site_extra" in ids
    power = next(g for g in resolved.groups if g.name == "power")
    assert any(r.id == "site_extra" for r in power.registers)


def test_map_level_disabled_entry_is_skipped(maps_dir: Path) -> None:
    _write(
        maps_dir,
        "disabled",
        BASE_MAP.replace(
            "scale: 0.01, asset: battery", "scale: 0.01, asset: battery, enabled: false"
        ),
    )
    resolved = resolve_vendor_map("disabled", {"asset_mappings": _bindings()})
    ids = {r.id for g in resolved.groups for r in g.registers}
    assert "bat_v" not in ids


# ── Inheritance (decision 5: explicit extends, bounded, no cross-file anchors) ──


CHILD_MAP = textwrap.dedent(
    """
    extends: base_map
    vendor: testvendor
    model_family: child
    source: synthetic fixture (unit test)
    verification: community
    groups:
      - name: power
        registers:
          - {id: bat_v, metric: battery.voltage_v, address: 587, type: uint16,
             scale: 0.1, asset: battery}
          - {id: pv3, metric: solar.mppt3_power_w, address: 674, type: uint16,
             scale: 10, asset: solar_array}
    assets: [inverter, battery, solar_array]
    registers_disable: [inv_state]
    """
)


def test_extends_merges_by_register_id(maps_dir: Path) -> None:
    _write(maps_dir, "child_map", CHILD_MAP)
    resolved = resolve_vendor_map(
        "child_map", {"asset_mappings": _bindings(solar_array=SOLAR)}
    )
    specs = {r.id: r for g in resolved.groups for r in g.registers}

    assert resolved.model_family == "child"
    assert resolved.protocol["baudrate"] == 9600  # inherited default
    assert specs["bat_v"].scale == pytest.approx(0.1)  # child override wins
    assert specs["pv3"].asset_id == SOLAR  # child addition
    assert "inv_state" not in specs  # child disabled a base entry
    assert "bat_chg" in specs  # untouched base group survives
    assert resolved.extends == ("base_map",)


def test_extends_depth_is_bounded_and_cycles_refused(maps_dir: Path) -> None:
    _write(maps_dir, "cyc_a", CHILD_MAP.replace("extends: base_map", "extends: cyc_b"))
    _write(maps_dir, "cyc_b", CHILD_MAP.replace("extends: base_map", "extends: cyc_a"))
    with pytest.raises(VendorMapError, match="cycle|depth"):
        load_vendor_map("cyc_a")

    prev = "base_map"
    for i in range(vm.MAX_EXTENDS_DEPTH + 1):
        name = f"deep_{i}"
        _write(
            maps_dir, name, CHILD_MAP.replace("extends: base_map", f"extends: {prev}")
        )
        prev = name
    with pytest.raises(VendorMapError, match="depth"):
        load_vendor_map(prev)


def test_extends_target_must_be_a_plain_name(maps_dir: Path) -> None:
    _write(
        maps_dir,
        "bad_ext",
        CHILD_MAP.replace("extends: base_map", "extends: ../base_map"),
    )
    with pytest.raises(VendorMapError, match="map name"):
        load_vendor_map("bad_ext")


# ── Validation of shapes ────────────────────────────────────────


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("address: 587", "address: 70000"),
        ("type: uint16,", "type: uint128,"),
        ("function_code: 3", "function_code: 6"),
        ("address_base: 0", "address_base: 2"),
        ("poll_interval_seconds: 30", "poll_interval_seconds: 0"),
    ],
)
def test_malformed_entries_refused(maps_dir: Path, mutation: str, message: str) -> None:
    _write(maps_dir, "malformed", BASE_MAP.replace(mutation, message))
    with pytest.raises(VendorMapError):
        load_vendor_map("malformed")


def test_duplicate_register_ids_refused(maps_dir: Path) -> None:
    _write(maps_dir, "dupe", BASE_MAP.replace("id: bat_chg", "id: bat_v"))
    with pytest.raises(VendorMapError, match="duplicate"):
        load_vendor_map("dupe")


def test_write_declarations_are_refused(maps_dir: Path) -> None:
    for key in ("write_mappings", "writes", "write_registers"):
        _write(
            maps_dir,
            "writer",
            BASE_MAP + f"\n{key}:\n  battery.soc_pct: {{address: 1}}\n",
        )
        with pytest.raises(VendorMapError, match="write"):
            load_vendor_map("writer")


def test_high_word_offset_widens_the_register_span(maps_dir: Path) -> None:
    _write(
        maps_dir,
        "split",
        BASE_MAP.replace(
            "type: uint32_le, scale: 0.1",
            "type: uint32_le, high_word_offset: 2, scale: 0.1",
        ),
    )
    spec = next(s for s in load_vendor_map("split").all_registers if s.id == "bat_chg")
    assert spec.high_word_offset == 2
    assert spec.word_count == 3  # 516, 517 (not ours), 518


@pytest.mark.parametrize(
    "replacement, message",
    [
        # Only the little-endian 32-bit types have a documented split case.
        ("max: 1000, high_word_offset: 2}", "high_word_offset"),
        # Offset 1 is just an ordinary adjacent pair: use the plain type.
        ("type: uint32_le, high_word_offset: 1,", "high_word_offset"),
        ("type: uint32_le, high_word_offset: 2, na: [4294967295],", "na"),
        ("type: uint32_le, high_word_offset: 2, words: 2,", "words"),
    ],
)
def test_high_word_offset_is_refused_where_it_cannot_be_meant(
    maps_dir: Path, replacement: str, message: str
) -> None:
    if replacement.startswith("max: 1000"):
        text = BASE_MAP.replace("max: 1000}", replacement)
    else:
        text = BASE_MAP.replace("type: uint32_le,", replacement)
    assert text != BASE_MAP
    _write(maps_dir, "badsplit", text)
    with pytest.raises(VendorMapError, match=message):
        load_vendor_map("badsplit")


def test_two_registers_emitting_one_metric_for_one_asset_are_refused(
    maps_dir: Path,
) -> None:
    """measurements is keyed (time, asset, metric): one of the two would vanish."""
    _write(
        maps_dir,
        "twoscale",
        BASE_MAP.replace(
            "features:\n  gen_port: false",
            "features:\n  gen_port: false\n  scale_a: false\n  scale_b: false",
        ).replace(
            "      - {id: gen_w,",
            "      - {id: v_a, metric: battery.current_a, address: 591, type: int16,\n"
            "         asset: battery, feature: scale_a}\n"
            "      - {id: v_b, metric: battery.current_a, address: 591, type: int16,\n"
            "         scale: 10, asset: battery, feature: scale_b}\n"
            "      - {id: gen_w,",
        ),
    )
    # Either alternative alone resolves.
    for feature in ("scale_a", "scale_b"):
        resolved = resolve_vendor_map(
            "twoscale", {"asset_mappings": _bindings(), "features": {feature: True}}
        )
        assert [s.metric for s in resolved.all_registers].count(
            "battery.current_a"
        ) == 1
    # Both at once is refused before anything is polled.
    with pytest.raises(VendorMapError, match="both emit"):
        resolve_vendor_map(
            "twoscale",
            {
                "asset_mappings": _bindings(),
                "features": {"scale_a": True, "scale_b": True},
            },
        )
    # Two roles bound to ONE asset collide just the same as one role twice.
    _write(
        maps_dir,
        "tworoles",
        BASE_MAP.replace(
            "      - {id: gen_w,",
            "      - {id: inv_v, metric: battery.voltage_v, address: 600, type: uint16,\n"
            "         asset: inverter}\n"
            "      - {id: gen_w,",
        ),
    )
    resolve_vendor_map("tworoles", {"asset_mappings": _bindings()})
    with pytest.raises(VendorMapError, match="both emit"):
        resolve_vendor_map(
            "tworoles", {"asset_mappings": {"inverter": BATTERY, "battery": BATTERY}}
        )
    # The probe binds nothing and reads every alternative side by side.
    probe = resolve_vendor_map("twoscale", {}, require_bindings=False)
    assert {"v_a", "v_b"} <= {s.id for s in probe.all_registers}


def test_a_negated_register_cannot_also_split_by_sign(maps_dir: Path) -> None:
    """The split reads the scaled value: -1 would swap import and export."""
    _write(
        maps_dir,
        "negsplit",
        BASE_MAP.replace(
            "      - {id: gen_w,",
            "      - {id: grid_w, metric: grid.import_power_w, address: 625, type: int16,\n"
            "         scale: -1, split_negative_to: grid.export_power_w, asset: inverter}\n"
            "      - {id: gen_w,",
        ),
    )
    with pytest.raises(VendorMapError, match="negative scale"):
        load_vendor_map("negsplit")


def test_multiword_register_must_fit_address_space(maps_dir: Path) -> None:
    _write(maps_dir, "edge", BASE_MAP.replace("address: 516", "address: 65535"))
    with pytest.raises(VendorMapError, match="address"):
        load_vendor_map("edge")


# ── Templates (ComAp) ───────────────────────────────────────────


def test_template_with_blank_addresses_is_inspectable_but_cannot_poll(
    maps_dir: Path,
) -> None:
    _write(
        maps_dir,
        "template",
        BASE_MAP.replace("address: 587", "address: null").replace(
            "address: 516", "address: ~"
        ),
    )
    loaded = load_vendor_map("template")
    assert loaded.complete is False
    assert set(loaded.missing_addresses) == {"bat_v", "bat_chg"}

    with pytest.raises(VendorMapError, match="template"):
        resolve_vendor_map("template", {"asset_mappings": _bindings()})


# ── Legacy config is untouched ──────────────────────────────────


def test_legacy_registers_config_has_no_vendor_map(maps_dir: Path) -> None:
    legacy = {
        "protocol": "tcp",
        "host": "1.2.3.4",
        "asset_id": "asset-legacy",
        "registers": [{"metric": "genset.power_w", "address": 100, "type": "uint16"}],
    }
    assert vm.vendor_map_name(legacy) is None
    assert vm.vendor_map_name({"vendor_map": "base_map"}) == "base_map"
    with pytest.raises(VendorMapError, match="registers"):
        resolve_vendor_map(
            "base_map",
            {"asset_mappings": _bindings(), "registers": legacy["registers"]},
        )


# ── Shipped maps: every file is community, sourced and resolvable ──


def _shipped_maps() -> list[Path]:
    return sorted(vm.DEFAULT_VENDOR_MAPS_DIR.glob("*.yaml"))


def test_shipped_maps_directory_exists() -> None:
    assert vm.DEFAULT_VENDOR_MAPS_DIR.is_dir()


@pytest.mark.parametrize("path", _shipped_maps(), ids=lambda p: p.stem)
def test_shipped_map_declares_community_and_pinned_sources(path: Path) -> None:
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    assert raw.get("verification") == "community", path.name
    assert isinstance(raw.get("source"), str) and raw["source"].strip(), path.name
    kind = raw.get("kind", "modbus_map")
    if kind == "sunspec_preset":
        return
    text = path.read_text().splitlines()
    for lineno, line in enumerate(text, start=1):
        stripped = line.strip()
        if "address:" in stripped and not stripped.startswith("#"):
            assert "# src:" in line, (
                f"{path.name}:{lineno} register line lacks '# src:'"
            )
    # A map may say it is NOT supported and NOT validated — that is the point
    # of the header. What it must never do is CLAIM either, so the scan looks
    # for claim-shaped phrases rather than the bare words, which a disclaimer
    # necessarily contains too.
    lowered = path.read_text().lower()
    for claim in (
        "is supported",
        "fully supported",
        "officially supported",
        "validated on a",
        "validated on the",
        "verification: bench",
        "verification: field",
    ):
        assert claim not in lowered, f"{path.name} claims {claim!r}"


@pytest.mark.parametrize("path", _shipped_maps(), ids=lambda p: p.stem)
def test_shipped_map_loads_with_the_production_loader(path: Path) -> None:
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    if raw.get("kind", "modbus_map") != "modbus_map":
        return
    loaded = load_vendor_map(path.stem)
    assert loaded.verification == "community"
    for group in loaded.groups:
        for spec in group.registers:
            assert spec.metric, f"{path.name}:{spec.id} has no metric"


@pytest.mark.parametrize("path", _shipped_maps(), ids=lambda p: p.stem)
def test_shipped_map_never_emits_one_metric_twice_per_asset(path: Path) -> None:
    """The duplicate-metric guard must not trip on any shipped map.

    Every role bound, then each feature enabled on its own: the widest
    configuration a site reaches without enabling two alternatives that a map
    declares as mutually exclusive.
    """
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    if raw.get("kind", "modbus_map") != "modbus_map":
        return
    loaded = load_vendor_map(path.stem)
    if not loaded.complete:
        return  # a template refuses to resolve for polling by design
    roles = loaded.required_roles + loaded.optional_roles
    bindings = {
        role: f"20000000-0000-0000-0000-{index:012d}"
        for index, role in enumerate(roles)
    }
    resolve_vendor_map(path.stem, {"asset_mappings": bindings})
    for feature in loaded.features:
        resolve_vendor_map(
            path.stem, {"asset_mappings": bindings, "features": {feature: True}}
        )
