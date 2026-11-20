"""Declarative vendor register maps for the generic Modbus adapter.

A vendor map (``vendor_maps/<name>.yaml``) describes how one family of
devices lays out its Modbus registers: protocol defaults, an identity
block, the asset ROLES the site must bind, and register groups with a
poll cadence each. The site YAML selects a map with ``vendor_map: <name>``
and binds roles with ``asset_mappings: {role: <asset uuid>}``; it may also
override transport settings and add or disable individual registers.

The loader is deliberately strict — a map that cannot be shown to be well
formed never reaches a serial bus:

* names are plain ``[a-z0-9_]`` identifiers resolved inside the shipped
  directory only (no traversal);
* ``extends: <name>`` is the ONLY inheritance mechanism (plan decision 5):
  resolved by this loader, bounded depth, cycles refused, ordinary YAML
  anchors are fine within a file but never cross files;
* ``verification`` must be one of ``community | bench | field`` and every
  map this tranche ships says ``community`` (spec §10);
* any write declaration is refused — a vendor map is read-only by
  construction;
* a map with blank addresses is a TEMPLATE (ComAp): it loads for
  inspection but refuses to resolve for polling.

Nothing here touches hardware or the cloud; it is pure configuration.
"""

from __future__ import annotations

import copy
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger()

DEFAULT_VENDOR_MAPS_DIR = Path(__file__).resolve().parent / "vendor_maps"
# Tests monkeypatch this to point at fixtures; production never changes it.
VENDOR_MAPS_DIR = DEFAULT_VENDOR_MAPS_DIR

MAX_EXTENDS_DEPTH = 3
_NAME_RE = re.compile(r"^[a-z0-9_]{1,64}$")

VERIFICATION_TIERS = ("community", "bench", "field")
_WRITE_KEYS = ("write_mappings", "writes", "write_registers", "write")
_FUNCTION_CODES = (3, 4)

# Word widths per decoder type. ``string``/``fault_bits``/``bitfield`` take an
# explicit ``words`` count (bounded below); everything else is fixed.
_FIXED_WORDS: dict[str, int] = {
    "uint16": 1,
    "int16": 1,
    "uint32": 2,
    "int32": 2,
    "uint32_le": 2,
    "int32_le": 2,
    "uint64": 4,
    "uint64_le": 4,
    "float32": 2,
    "float32_le": 2,
    "enum": 1,
}
_VARIABLE_WORDS: dict[str, tuple[int, int, int]] = {
    # type: (default, min, max)
    "string": (1, 1, 64),
    "bitfield": (1, 1, 2),
    "fault_bits": (1, 1, 8),
}
VALID_TYPES = frozenset(_FIXED_WORDS) | frozenset(_VARIABLE_WORDS)
# Types that may declare ``high_word_offset`` (a high word not adjacent to the
# low word). Little-endian only: the one documented case is Deye single-phase.
_SPLIT_WORD_TYPES = frozenset({"uint32_le", "int32_le"})
IDENTITY_ONLY_TYPES = frozenset({"string"})

_SITE_PROTOCOL_KEYS = (
    "protocol",
    "host",
    "port",
    "baudrate",
    "parity",
    "stopbits",
    "bytesize",
    "unit_id",
    "timeout_ms",
    "retries",
    "request_spacing_ms",
    "address_base",
    "max_gap",
    "max_block_words",
)


class VendorMapError(ValueError):
    """A vendor map or its site overrides are not safe to poll."""


@dataclass(frozen=True, slots=True)
class RegisterSpec:
    """One resolved register entry.

    ``address`` is the address AS PRINTED in the map (the map's
    ``address_base`` says whether that is 0- or 1-based); the Modbus engine
    subtracts the base once, at read time. ``asset_id`` is empty until the
    site binds the role.
    """

    id: str
    metric: str
    address: int
    data_type: str
    word_count: int
    asset_role: str
    group: str
    function_code: int
    unit_id: int
    scale: float = 1.0
    offset: float = 0.0
    asset_id: str = ""
    min_value: float | None = None
    max_value: float | None = None
    mask: int | None = None
    enum_values: dict[int, float] | None = None
    split_negative_to: str | None = None
    na_values: tuple[int, ...] = ()
    derived: str | None = None
    components: tuple[str, ...] = ()
    feature: str | None = None
    notes: str = ""
    # A 32-bit value whose HIGH word does not sit next to its low word. Deye's
    # single-phase lifetime grid-import counter is low word at 78, high word
    # at 80, with grid frequency in between. The register then spans
    # ``high_word_offset + 1`` words, read in ONE request so the two halves
    # come from the same instant; only the first and the offset word decode.
    high_word_offset: int | None = None

    @property
    def identity_only(self) -> bool:
        return self.data_type in IDENTITY_ONLY_TYPES


@dataclass(frozen=True, slots=True)
class RegisterGroup:
    name: str
    poll_interval_seconds: int
    registers: tuple[RegisterSpec, ...]


@dataclass(frozen=True, slots=True)
class VendorMap:
    """A loaded (and optionally site-resolved) vendor map."""

    name: str
    vendor: str
    model_family: str
    verification: str
    source: str
    protocol: dict[str, Any]
    identity: tuple[RegisterSpec, ...]
    groups: tuple[RegisterGroup, ...]
    required_roles: tuple[str, ...]
    optional_roles: tuple[str, ...]
    features: dict[str, bool]
    address_base: int
    complete: bool
    missing_addresses: tuple[str, ...]
    extends: tuple[str, ...] = ()
    notes: str = ""
    dropped_roles: tuple[str, ...] = ()
    resolved: bool = False

    @property
    def all_registers(self) -> tuple[RegisterSpec, ...]:
        return tuple(spec for group in self.groups for spec in group.registers)

    def to_summary(self) -> dict[str, Any]:
        """A JSON-safe description for evidence records and CLI output."""
        return {
            "name": self.name,
            "vendor": self.vendor,
            "model_family": self.model_family,
            "verification": self.verification,
            "source": self.source,
            "extends": list(self.extends),
            "address_base": self.address_base,
            "complete": self.complete,
            "missing_addresses": list(self.missing_addresses),
            "roles": {
                "required": list(self.required_roles),
                "optional": list(self.optional_roles),
            },
            "groups": [
                {
                    "name": group.name,
                    "poll_interval_seconds": group.poll_interval_seconds,
                    "registers": len(group.registers),
                }
                for group in self.groups
            ],
        }


# ── Public API ──────────────────────────────────────────────────


def vendor_map_name(config: dict[str, Any]) -> str | None:
    """The ``vendor_map`` a site config selects, or None for legacy YAML."""
    name = config.get("vendor_map")
    if name is None:
        return None
    if not isinstance(name, str):
        raise VendorMapError("vendor_map must be a map name string")
    return name


def load_sunspec_preset(name: str) -> dict[str, Any]:
    """Load a SunSpec vendor preset (``kind: sunspec_preset``).

    A preset is not a register map — the SunSpec adapter discovers the model
    chain itself. It carries the connection facts an installer would otherwise
    look up (SMA's unit 126, SolarEdge's port 1502) and the exclusions that
    stop it being pointed at hardware it cannot read.
    """
    _validate_name(name)
    raw = _read_yaml(name)
    if raw.get("kind") != "sunspec_preset":
        raise VendorMapError(f"{name!r} is not a SunSpec preset")
    if raw.get("verification") not in VERIFICATION_TIERS:
        raise VendorMapError(f"SunSpec preset {name!r}: verification must be declared")
    if not str(raw.get("source", "")).strip():
        raise VendorMapError(f"SunSpec preset {name!r}: source must be declared")
    connection = raw.get("connection")
    if not isinstance(connection, dict):
        raise VendorMapError(f"SunSpec preset {name!r}: connection must be a mapping")
    for key in _WRITE_KEYS:
        if key in raw or key in connection:
            raise VendorMapError(
                f"SunSpec preset {name!r} declares {key!r}: presets are read-only"
            )
    return raw


def load_vendor_map(name: str) -> VendorMap:
    """Load and validate a shipped map without binding it to a site.

    Templates load (``complete=False``) so a technician can inspect them.
    """
    raw, chain = _load_raw(name, depth=0, visited=())
    return _build(name, raw, chain, site=None)


def resolve_vendor_map(
    name: str, site_config: dict[str, Any], *, require_bindings: bool = True
) -> VendorMap:
    """Load a map and apply the site's overrides and role bindings.

    Refuses templates, legacy ``registers:`` mixed into a map-driven config,
    unbound required roles and non-UUID bindings.

    ``require_bindings=False`` is the validation probe: it reads registers to
    produce evidence and never emits telemetry, so it has no assets to bind.
    Inventing asset UUIDs to satisfy the resolver would put fabricated
    identifiers into an evidence record, which is worse than having none.
    """
    if site_config.get("registers"):
        raise VendorMapError(
            "a vendor_map config cannot also carry a legacy registers: list — "
            "use registers_add / registers_disable"
        )
    raw, chain = _load_raw(name, depth=0, visited=())
    return _build(name, raw, chain, site=site_config, require_bindings=require_bindings)


# ── Raw loading and inheritance ─────────────────────────────────


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise VendorMapError(
            f"invalid vendor map name {name!r}: use lowercase letters, digits and "
            "underscores only"
        )
    return name


def _read_yaml(name: str) -> dict[str, Any]:
    path = (VENDOR_MAPS_DIR / f"{name}.yaml").resolve()
    if path.parent != VENDOR_MAPS_DIR.resolve():
        raise VendorMapError(f"invalid vendor map name {name!r}")
    if not path.is_file():
        raise VendorMapError(f"vendor map {name!r} not found under {VENDOR_MAPS_DIR}")
    with open(path, encoding="utf-8") as handle:
        try:
            data = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise VendorMapError(
                f"vendor map {name!r} is not valid YAML: {exc}"
            ) from exc
    if not isinstance(data, dict):
        raise VendorMapError(f"vendor map {name!r} must be a mapping at the top level")
    return data


def _load_raw(
    name: str, *, depth: int, visited: tuple[str, ...]
) -> tuple[dict[str, Any], tuple[str, ...]]:
    _validate_name(name)
    if name in visited:
        raise VendorMapError(
            f"vendor map inheritance cycle: {' -> '.join(visited + (name,))}"
        )
    if depth > MAX_EXTENDS_DEPTH:
        raise VendorMapError(
            f"vendor map inheritance depth exceeds {MAX_EXTENDS_DEPTH} at {name!r}"
        )
    raw = _read_yaml(name)
    for key in _WRITE_KEYS:
        if key in raw:
            raise VendorMapError(
                f"vendor map {name!r} declares {key!r}: maps are read-only"
            )
    parent = raw.get("extends")
    if parent is None:
        return raw, ()
    if not isinstance(parent, str):
        raise VendorMapError(f"vendor map {name!r}: extends must name one map")
    base, base_chain = _load_raw(parent, depth=depth + 1, visited=visited + (name,))
    merged = _merge_raw(base, raw)
    return merged, (parent, *base_chain)


def _merge_raw(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in child.items():
        if key == "extends":
            continue
        if key in ("protocol_defaults", "identity", "features") and isinstance(
            value, dict
        ):
            current = merged.get(key)
            merged[key] = {**(current if isinstance(current, dict) else {}), **value}
        elif key == "groups":
            merged["groups"] = _merge_groups(merged.get("groups"), value)
        elif key == "registers_disable":
            existing = merged.get("registers_disable") or []
            merged["registers_disable"] = list(existing) + list(value or [])
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _merge_groups(base_groups: Any, child_groups: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [copy.deepcopy(g) for g in (base_groups or [])]
    by_name = {g.get("name"): g for g in result if isinstance(g, dict)}
    for child in child_groups or []:
        if not isinstance(child, dict):
            raise VendorMapError("groups entries must be mappings")
        target = by_name.get(child.get("name"))
        if target is None:
            result.append(copy.deepcopy(child))
            by_name[child.get("name")] = result[-1]
            continue
        for key, value in child.items():
            if key == "registers":
                target["registers"] = _merge_registers(target.get("registers"), value)
            else:
                target[key] = copy.deepcopy(value)
    return result


def _merge_registers(base_regs: Any, child_regs: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [copy.deepcopy(r) for r in (base_regs or [])]
    index = {r.get("id"): i for i, r in enumerate(result) if isinstance(r, dict)}
    for entry in child_regs or []:
        if not isinstance(entry, dict):
            raise VendorMapError("registers entries must be mappings")
        rid = entry.get("id")
        if rid in index:
            result[index[rid]] = copy.deepcopy(entry)
        else:
            result.append(copy.deepcopy(entry))
            index[rid] = len(result) - 1
    return result


# ── Building the validated map ──────────────────────────────────


def _build(
    name: str,
    raw: dict[str, Any],
    chain: tuple[str, ...],
    *,
    site: dict[str, Any] | None,
    require_bindings: bool = True,
) -> VendorMap:
    vendor = _req_str(raw, "vendor", name)
    model_family = _req_str(raw, "model_family", name)
    source = _req_str(raw, "source", name)
    verification = raw.get("verification")
    if verification not in VERIFICATION_TIERS:
        raise VendorMapError(
            f"vendor map {name!r}: verification must be one of {VERIFICATION_TIERS}, "
            f"got {verification!r}"
        )
    notes = str(raw.get("notes") or "")

    protocol = _protocol(raw.get("protocol_defaults"), name)
    if site is not None:
        for key in _SITE_PROTOCOL_KEYS:
            if key in site:
                protocol[key] = site[key]
    address_base = _int_in(protocol.get("address_base", 0), 0, 1, "address_base", name)
    protocol["address_base"] = address_base
    default_fc = _int_choice(protocol.get("function_code", 3), _FUNCTION_CODES, name)
    protocol["function_code"] = default_fc
    default_unit = _int_in(protocol.get("unit_id", 1), 0, 247, "unit_id", name)
    protocol["unit_id"] = default_unit

    required_roles = _roles(raw.get("assets"), name, "assets")
    optional_roles = _roles(raw.get("optional_assets"), name, "optional_assets")
    if not required_roles:
        raise VendorMapError(f"vendor map {name!r}: assets must list at least one role")
    all_roles = set(required_roles) | set(optional_roles)

    features: dict[str, bool] = {}
    for key, value in (raw.get("features") or {}).items():
        features[str(key)] = bool(value)
    if site is not None:
        for key, value in (site.get("features") or {}).items():
            if str(key) not in features:
                raise VendorMapError(f"vendor map {name!r}: unknown feature {key!r}")
            features[str(key)] = bool(value)

    disabled = {str(x) for x in (raw.get("registers_disable") or [])}
    if site is not None:
        disabled |= {str(x) for x in (site.get("registers_disable") or [])}

    bindings = _bindings(site, name) if site is not None else {}
    if site is not None and require_bindings:
        missing = [role for role in required_roles if role not in bindings]
        if missing:
            raise VendorMapError(
                f"vendor map {name!r}: asset_mappings must bind required roles "
                f"{missing} (site YAML: asset_mappings: {{role: <asset uuid>}})"
            )
    dropped_roles = tuple(
        role for role in optional_roles if site is not None and role not in bindings
    )

    identity = tuple(
        _register(
            {**entry, "id": entry.get("id", key)},
            group="identity",
            group_fc=default_fc,
            unit_default=default_unit,
            roles=all_roles | {"identity"},
            address_base=address_base,
            name=name,
            identity=True,
        )
        for key, entry in (raw.get("identity") or {}).items()
    )

    groups: list[RegisterGroup] = []
    seen_ids: set[str] = set()
    missing_addresses: list[str] = []
    raw_groups = raw.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise VendorMapError(f"vendor map {name!r}: groups must be a non-empty list")
    site_adds = list((site or {}).get("registers_add") or [])
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            raise VendorMapError(
                f"vendor map {name!r}: groups entries must be mappings"
            )
        gname = _req_str(raw_group, "name", name)
        interval = _int_in(
            raw_group.get(
                "poll_interval_seconds", protocol.get("poll_interval_seconds", 60)
            ),
            1,
            86400,
            f"groups[{gname}].poll_interval_seconds",
            name,
        )
        group_fc = _int_choice(
            raw_group.get("function_code", default_fc), _FUNCTION_CODES, name
        )
        entries = list(raw_group.get("registers") or [])
        entries.extend(e for e in site_adds if str(e.get("group", "")) == gname)
        specs: list[RegisterSpec] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise VendorMapError(
                    f"vendor map {name!r}: registers entries must be mappings"
                )
            rid = str(entry.get("id") or "")
            if not rid:
                raise VendorMapError(
                    f"vendor map {name!r}: every register needs a stable id "
                    f"(group {gname!r}, entry {entry})"
                )
            if rid in seen_ids:
                raise VendorMapError(
                    f"vendor map {name!r}: duplicate register id {rid!r}"
                )
            seen_ids.add(rid)
            if entry.get("enabled", True) is False or rid in disabled:
                continue
            if entry.get("address") is None and entry.get("derived") is None:
                missing_addresses.append(rid)
                continue
            spec = _register(
                entry,
                group=gname,
                group_fc=group_fc,
                unit_default=default_unit,
                roles=all_roles,
                address_base=address_base,
                name=name,
                identity=False,
            )
            if (
                spec.feature is not None
                and require_bindings
                and not features.get(spec.feature, False)
            ):
                continue
            if site is not None:
                asset_id = bindings.get(spec.asset_role, "")
                if not asset_id and require_bindings:
                    continue  # optional role left unbound — registers dropped
                spec = _with_asset(spec, asset_id, protocol["unit_id"], entry)
            specs.append(spec)
        groups.append(RegisterGroup(gname, interval, tuple(specs)))

    unknown_adds = [
        e.get("id")
        for e in site_adds
        if str(e.get("group", "")) not in {g.name for g in groups}
    ]
    if unknown_adds:
        raise VendorMapError(
            f"vendor map {name!r}: registers_add names unknown group for {unknown_adds}"
        )

    _validate_components(groups, seen_ids, name)
    if site is not None and require_bindings:
        _refuse_duplicate_metrics(groups, name)

    complete = not missing_addresses
    if site is not None and not complete:
        raise VendorMapError(
            f"vendor map {name!r} is a template: addresses are blank for "
            f"{missing_addresses} — fill them from the controller's exported "
            "table before polling"
        )

    return VendorMap(
        name=name,
        vendor=vendor,
        model_family=model_family,
        verification=str(verification),
        source=source,
        protocol=protocol,
        identity=identity,
        groups=tuple(groups),
        required_roles=required_roles,
        optional_roles=optional_roles,
        features=features,
        address_base=address_base,
        complete=complete,
        missing_addresses=tuple(missing_addresses),
        extends=chain,
        notes=notes,
        dropped_roles=dropped_roles,
        resolved=site is not None,
    )


def _with_asset(
    spec: RegisterSpec, asset_id: str, unit_default: int, entry: dict[str, Any]
) -> RegisterSpec:
    # A site-level unit_id override applies to every register that did not
    # pin its own unit id in the map.
    unit_id = spec.unit_id if "unit_id" in entry else unit_default
    return RegisterSpec(
        **{
            **{f: getattr(spec, f) for f in RegisterSpec.__dataclass_fields__},
            "asset_id": asset_id,
            "unit_id": unit_id,
        }
    )


def _register(
    entry: dict[str, Any],
    *,
    group: str,
    group_fc: int,
    unit_default: int,
    roles: set[str],
    address_base: int,
    name: str,
    identity: bool,
) -> RegisterSpec:
    rid = str(entry.get("id") or "")
    data_type = str(entry.get("type", "uint16"))
    if data_type not in VALID_TYPES:
        raise VendorMapError(
            f"vendor map {name!r}: register {rid!r} has unknown type {data_type!r} "
            f"(valid: {sorted(VALID_TYPES)})"
        )
    high_word_offset: int | None = None
    if entry.get("high_word_offset") is not None:
        if data_type not in _SPLIT_WORD_TYPES:
            raise VendorMapError(
                f"vendor map {name!r}: register {rid!r} high_word_offset applies only to "
                f"{sorted(_SPLIT_WORD_TYPES)}"
            )
        high_word_offset = _int_in(
            entry["high_word_offset"], 2, 8, f"{rid}.high_word_offset", name
        )
        if entry.get("na") is not None:
            # A sentinel is a bit pattern across contiguous words; across a
            # gap there is no documented pattern to compare against.
            raise VendorMapError(
                f"vendor map {name!r}: register {rid!r} cannot declare na with "
                "high_word_offset"
            )
    if high_word_offset is not None:
        words = high_word_offset + 1
        if "words" in entry and int(entry["words"]) != words:
            raise VendorMapError(
                f"vendor map {name!r}: register {rid!r} words conflicts with "
                "high_word_offset"
            )
    elif data_type in _FIXED_WORDS:
        words = _FIXED_WORDS[data_type]
        if "words" in entry and int(entry["words"]) != words:
            raise VendorMapError(
                f"vendor map {name!r}: register {rid!r} words conflicts with type"
            )
    else:
        default, lo, hi = _VARIABLE_WORDS[data_type]
        words = _int_in(entry.get("words", default), lo, hi, f"{rid}.words", name)

    derived = entry.get("derived")
    if derived is not None and derived != "sum":
        raise VendorMapError(
            f"vendor map {name!r}: register {rid!r} derived must be 'sum'"
        )
    components = tuple(str(c) for c in (entry.get("of") or ()))
    if derived == "sum" and len(components) < 2:
        raise VendorMapError(
            f"vendor map {name!r}: derived sum {rid!r} needs >= 2 components"
        )

    if derived is None:
        address = _int_in(entry.get("address"), 0, 65535, f"{rid}.address", name)
        effective = address - address_base
        if effective < 0 or effective + words > 65536:
            raise VendorMapError(
                f"vendor map {name!r}: register {rid!r} address {address} (+{words} words, "
                f"base {address_base}) leaves the Modbus address space"
            )
    else:
        address = -1

    metric = str(entry.get("metric") or "")
    if identity:
        if not metric:
            metric = f"identity.{rid}"
    elif not metric or "." not in metric:
        raise VendorMapError(
            f"vendor map {name!r}: register {rid!r} needs a canonical metric"
        )
    if not identity and data_type in IDENTITY_ONLY_TYPES:
        raise VendorMapError(
            f"vendor map {name!r}: register {rid!r} type {data_type!r} is identity-only and "
            "cannot be a telemetry metric"
        )

    role = str(entry.get("asset") or ("identity" if identity else ""))
    if role not in roles:
        raise VendorMapError(
            f"vendor map {name!r}: register {rid!r} names role {role!r} which is not in "
            f"assets/optional_assets {sorted(roles - {'identity'})}"
        )

    function_code = _int_choice(
        entry.get("function_code", group_fc), _FUNCTION_CODES, name
    )
    unit_id = _int_in(
        entry.get("unit_id", unit_default), 0, 247, f"{rid}.unit_id", name
    )
    scale = _float(entry.get("scale", 1.0), f"{rid}.scale", name)
    if scale == 0:
        raise VendorMapError(f"vendor map {name!r}: register {rid!r} scale cannot be 0")
    offset = _float(entry.get("offset", 0.0), f"{rid}.offset", name)
    min_value = _opt_float(entry.get("min"), f"{rid}.min", name)
    max_value = _opt_float(entry.get("max"), f"{rid}.max", name)
    if min_value is not None and max_value is not None and min_value > max_value:
        raise VendorMapError(f"vendor map {name!r}: register {rid!r} min exceeds max")

    mask: int | None = None
    if data_type == "bitfield":
        mask = _int_in(
            entry.get("mask"), 1, (1 << (16 * words)) - 1, f"{rid}.mask", name
        )

    enum_values: dict[int, float] | None = None
    if data_type == "enum":
        table = entry.get("values")
        if not isinstance(table, dict) or not table:
            raise VendorMapError(
                f"vendor map {name!r}: enum register {rid!r} needs values"
            )
        enum_values = {}
        for key, value in table.items():
            enum_values[_int_in(key, 0, 65535, f"{rid}.values", name)] = _float(
                value, f"{rid}.values[{key}]", name
            )

    na_raw = entry.get("na")
    na_candidates = (
        na_raw
        if isinstance(na_raw, (list, tuple))
        else ([] if na_raw is None else [na_raw])
    )
    width_bits = 16 * words
    na_values = tuple(
        _int_in(candidate, 0, (1 << width_bits) - 1, f"{rid}.na", name)
        for candidate in na_candidates
    )

    split = entry.get("split_negative_to")
    if split is not None and (not isinstance(split, str) or "." not in split):
        raise VendorMapError(
            f"vendor map {name!r}: register {rid!r} split_negative_to needs a metric"
        )
    if split is not None and not data_type.startswith("int"):
        raise VendorMapError(
            f"vendor map {name!r}: register {rid!r} split_negative_to requires a signed type"
        )
    if split is not None and scale < 0:
        # The split reads the SCALED value, so a negated register would send
        # import to export and export to import without any visible error.
        raise VendorMapError(
            f"vendor map {name!r}: register {rid!r} cannot combine split_negative_to "
            "with a negative scale — swap metric and split_negative_to instead"
        )

    feature = entry.get("feature")
    return RegisterSpec(
        id=rid,
        metric=metric,
        address=address,
        data_type=data_type,
        word_count=words,
        asset_role=role,
        group=group,
        function_code=function_code,
        unit_id=unit_id,
        scale=scale,
        offset=offset,
        min_value=min_value,
        max_value=max_value,
        mask=mask,
        enum_values=enum_values,
        split_negative_to=split,
        na_values=na_values,
        derived=derived,
        components=components,
        feature=str(feature) if feature is not None else None,
        notes=str(entry.get("notes") or ""),
        high_word_offset=high_word_offset,
    )


def _validate_components(
    groups: list[RegisterGroup], declared_ids: set[str], name: str
) -> None:
    """A derived sum must name registers this map declares.

    ``declared_ids`` deliberately includes entries that were disabled or
    dropped by a feature flag: a component absent from THIS resolution is
    normal (an HP3 map's PV3/PV4, a gen port left off), and the decode engine
    withholds the sum when any component did not decode this poll. What is
    never acceptable is a component id that no register in the map defines —
    that is a typo, and it would silently make the sum unreachable forever.
    """
    for group in groups:
        for spec in group.registers:
            if spec.derived != "sum":
                continue
            if spec.id in spec.components:
                raise VendorMapError(
                    f"vendor map {name!r}: register {spec.id!r} sums itself"
                )
            unknown = [c for c in spec.components if c not in declared_ids]
            if unknown:
                raise VendorMapError(
                    f"vendor map {name!r}: derived register {spec.id!r} sums unknown "
                    f"register ids {unknown}"
                )


def _refuse_duplicate_metrics(groups: list[RegisterGroup], name: str) -> None:
    """Two emitted registers may never land on one (metric, asset).

    ``measurements`` is keyed on (time, asset, metric) and both edge write
    paths are ``INSERT OR IGNORE``, so the second of two same-instant readings
    would vanish without a trace. This is reachable when a map offers
    mutually exclusive alternatives behind features (one register at two
    candidate scales) and a site enables both: refuse at resolve time rather
    than let one of them silently disappear on every poll.
    """
    owners: dict[tuple[str, str], str] = {}
    for group in groups:
        for spec in group.registers:
            emitted = [spec.metric]
            if spec.split_negative_to is not None:
                emitted.append(spec.split_negative_to)
            for metric in emitted:
                # Keyed on the bound asset, not the role: two roles bound to
                # one asset UUID collide on the measurements key just the same.
                key = (metric, spec.asset_id or spec.asset_role)
                other = owners.get(key)
                if other is not None:
                    raise VendorMapError(
                        f"vendor map {name!r}: registers {other!r} and {spec.id!r} both "
                        f"emit {metric!r} for role {spec.asset_role!r} on one asset — "
                        "enable only one of them (check the map's features)"
                    )
                owners[key] = spec.id


# ── Small validators ────────────────────────────────────────────


def _req_str(raw: dict[str, Any], key: str, name: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VendorMapError(f"vendor map {name!r}: {key!r} must be a non-empty string")
    return value.strip()


def _protocol(raw: Any, name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise VendorMapError(
            f"vendor map {name!r}: protocol_defaults must be a mapping"
        )
    protocol = dict(raw)
    if "protocol" in protocol and protocol["protocol"] not in ("tcp", "rtu"):
        raise VendorMapError(f"vendor map {name!r}: protocol must be tcp or rtu")
    if "parity" in protocol and str(protocol["parity"]).upper() not in ("N", "E", "O"):
        raise VendorMapError(f"vendor map {name!r}: parity must be N, E or O")
    if "stopbits" in protocol:
        _int_choice(protocol["stopbits"], (1, 2), name)
    if "bytesize" in protocol:
        _int_choice(protocol["bytesize"], (7, 8), name)
    return protocol


def _roles(raw: Any, name: str, key: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(r, str) and r for r in raw):
        raise VendorMapError(f"vendor map {name!r}: {key} must be a list of role names")
    if len(set(raw)) != len(raw):
        raise VendorMapError(f"vendor map {name!r}: {key} repeats a role")
    return tuple(raw)


def _bindings(site: dict[str, Any], name: str) -> dict[str, str]:
    mappings = site.get("asset_mappings") or {}
    if not isinstance(mappings, dict):
        raise VendorMapError(f"vendor map {name!r}: asset_mappings must be a mapping")
    bound: dict[str, str] = {}
    for role, asset in mappings.items():
        if asset in (None, ""):
            continue
        try:
            uuid.UUID(str(asset))
        except (AttributeError, TypeError, ValueError) as exc:
            raise VendorMapError(
                f"vendor map {name!r}: asset_mappings[{role!r}] must be a UUID, got {asset!r}"
            ) from exc
        bound[str(role)] = str(asset)
    return bound


def _int_in(value: Any, lo: int, hi: int, label: str, name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise VendorMapError(f"vendor map {name!r}: {label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise VendorMapError(
            f"vendor map {name!r}: {label} must be an integer"
        ) from exc
    if isinstance(value, float) and number != value:
        raise VendorMapError(f"vendor map {name!r}: {label} must be a whole number")
    if not lo <= number <= hi:
        raise VendorMapError(
            f"vendor map {name!r}: {label}={number} outside {lo}..{hi}"
        )
    return number


def _int_choice(value: Any, choices: tuple[int, ...], name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise VendorMapError(f"vendor map {name!r}: expected one of {choices}") from exc
    if number not in choices:
        raise VendorMapError(f"vendor map {name!r}: {number} is not one of {choices}")
    return number


def _float(value: Any, label: str, name: str) -> float:
    if isinstance(value, bool):
        raise VendorMapError(f"vendor map {name!r}: {label} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VendorMapError(f"vendor map {name!r}: {label} must be a number") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise VendorMapError(f"vendor map {name!r}: {label} must be finite")
    return number


def _opt_float(value: Any, label: str, name: str) -> float | None:
    return None if value is None else _float(value, label, name)
