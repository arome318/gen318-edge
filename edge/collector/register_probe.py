"""Opt-in Modbus register probe — mapping evidence for lifetime counters.

Sweeps configured register ranges on a Victron GX (or any Modbus TCP
server), records every plausible decoding of each register — raw u16,
u32 big-endian, and u32 little-endian, each at scales 1/10/100 — into the
local ``register_candidates`` SQLite table, and ships ONE tier-3 summary
event per cycle. Read-only: the probe never writes a Modbus register and
never writes to ``measurements``.

Honesty notes (verified against Victron dbus_modbustcp attributes.csv):

- Unit 225 regs 301/302 (/History/DischargedEnergy, /History/ChargedEnergy)
  are uint16 at scale 10 — they CAP at 6553.5 kWh and are user-resettable
  via VictronConnect. Candidates from them are evidence, not lifetime truth.
- ``solar.total_yield_kwh`` (unit 247 reg 790) is currently mapped uint16
  scale 10 in ``victron_modbus.py`` and silently wraps at 6553.5 kWh — a
  known defect this probe helps confirm (add ``247:790-791`` to
  ``EDGE_REGISTER_PROBE_RANGES`` to observe 790/791 as a u32 pair).
- Victron 32-bit values are big-endian word order (lower address = most
  significant word; mappings.cpp: ``offset == 0 → v >> 16``), but the probe
  records BOTH word orders so the evidence itself decides.
- Candidate data feeds a mapping DECISION by a human. It never changes a
  telemetry metric silently.

Offline-first: pymodbus absent, host unset, device unreachable, or any
per-cycle error all degrade to a logged no-op — nothing here may fail or
block telemetry collection (edge invariant 1).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from datetime import UTC, datetime
from typing import Any

import structlog

from common.database import EdgeDatabase

logger = structlog.get_logger()

DEFAULT_RANGES = "225:280-310,225:325-335"
RETENTION_HOURS = 168  # 7 days of candidate evidence

_MAX_READ_REGISTERS = 120  # chunk size, mirrors sunspec_modbus._read_words
_MAX_RANGE_LENGTH = 512  # RAM guard — collector runs under a 256m limit
_SCALES = (1, 10, 100)
_SCALE_ONE_DECODINGS = frozenset({"u16", "u32be", "u32le"})
_SUMMARY_TOP_N = 20


def parse_ranges(spec: str) -> list[tuple[int, int, int]]:
    """Parse ``"unit:start-end,unit:addr,..."`` into (unit, start, end) tuples.

    Invalid entries are logged and skipped — a bad config value must never
    take the collector down.
    """
    ranges: list[tuple[int, int, int]] = []
    for raw_entry in spec.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        try:
            unit_part, _, addr_part = entry.partition(":")
            unit_id = int(unit_part)
            if "-" in addr_part:
                start_str, _, end_str = addr_part.partition("-")
                start, end = int(start_str), int(end_str)
            else:
                start = end = int(addr_part)
        except ValueError:
            logger.warning("register_probe_bad_range", entry=entry)
            continue
        if not (0 <= unit_id <= 255 and 0 <= start <= end <= 65535):
            logger.warning("register_probe_bad_range", entry=entry)
            continue
        if end - start + 1 > _MAX_RANGE_LENGTH:
            logger.warning(
                "register_probe_range_too_large",
                entry=entry,
                max_registers=_MAX_RANGE_LENGTH,
            )
            continue
        ranges.append((unit_id, start, end))
    return ranges


class RegisterProbe:
    """Read-only register sweeper producing decoding candidates.

    Owns its own Modbus TCP client (never shares the adapter's socket),
    paces reads politely (>= 1s between chunks by default — the GX is
    Pi-class hardware), and persists candidates idempotently.
    """

    def __init__(
        self,
        db: EdgeDatabase,
        host: str,
        port: int = 502,
        ranges: str = DEFAULT_RANGES,
        interval_seconds: int = 3600,
        pace_seconds: float = 1.0,
        retention_hours: int = RETENTION_HOURS,
    ) -> None:
        self.db = db
        self._host = host
        self._port = port
        self._ranges_spec = ranges
        self._ranges = parse_ranges(ranges)
        self._interval = max(1, int(interval_seconds))
        self._pace_seconds = pace_seconds
        self._retention_ms = int(retention_hours) * 3600 * 1000

        # Guarded optional dep (inference_engine idiom): the collector must
        # construct and run on machines without pymodbus.
        self._client_cls: Any = None
        try:
            from pymodbus.client import AsyncModbusTcpClient

            self._client_cls = AsyncModbusTcpClient
        except ImportError:
            logger.warning(
                "register_probe_pymodbus_missing",
                msg="pymodbus not installed — register probe is a no-op",
            )

    async def run(self) -> None:
        """Supervised loop. Never raises; degrades to a logged no-op."""
        if self._client_cls is None:
            logger.warning("register_probe_disabled", reason="pymodbus_missing")
        elif not self._host:
            logger.warning("register_probe_disabled", reason="no_host_configured")
        elif not self._ranges:
            logger.warning(
                "register_probe_disabled",
                reason="no_valid_ranges",
                spec=self._ranges_spec,
            )
        while True:
            try:
                await self._run_cycle()
            except Exception:
                # Edge invariant 1: probe failures never propagate.
                logger.exception("register_probe_cycle_failed")
            await asyncio.sleep(self._interval)

    async def _run_cycle(self, now_ms: int | None = None) -> None:
        """One sweep: read ranges, persist candidates, prune, enqueue summary."""
        if self._client_cls is None or not self._host or not self._ranges:
            return
        cycle_ts = now_ms if now_ms is not None else int(time.time() * 1000)

        rows: list[tuple[int, int, int, str, float]] = []
        client = self._client_cls(host=self._host, port=self._port, timeout=3.0)
        try:
            connected = await client.connect()
            if not connected:
                logger.warning(
                    "register_probe_connect_failed",
                    host=self._host,
                    port=self._port,
                )
                return
            for index, (unit_id, start, end) in enumerate(self._ranges):
                if index and self._pace_seconds:
                    await asyncio.sleep(self._pace_seconds)
                words = await self._read_range(client, unit_id, start, end)
                if words is None:
                    continue
                rows.extend(self._decode_candidates(cycle_ts, unit_id, start, words))
        finally:
            client.close()

        if not rows:
            return
        self.db.insert_register_candidates(rows)
        self.db.prune_register_candidates(cycle_ts - self._retention_ms)
        self._enqueue_summary(cycle_ts, rows)
        logger.info(
            "register_probe_cycle_complete",
            candidate_rows=len(rows),
            ranges=self._ranges_spec,
        )

    async def _read_range(
        self, client: Any, unit_id: int, start: int, end: int
    ) -> list[int] | None:
        """Batched contiguous read; None on any failure (range is skipped)."""
        words: list[int] = []
        cursor = start
        remaining = end - start + 1
        while remaining:
            if words and self._pace_seconds:
                await asyncio.sleep(self._pace_seconds)
            chunk = min(remaining, _MAX_READ_REGISTERS)
            # Imported lazily, like every other collector import in this
            # module, so a machine without pymodbus still loads the probe and
            # degrades to the logged no-op rather than failing at import.
            from collector.adapters.generic_modbus import DEVICE_KWARG

            try:
                result = await client.read_holding_registers(
                    address=cursor,
                    count=chunk,
                    **{DEVICE_KWARG: unit_id},
                )
            except Exception:
                logger.warning(
                    "register_probe_read_error", unit_id=unit_id, address=cursor
                )
                return None
            if result.isError() or len(result.registers) != chunk:
                logger.warning(
                    "register_probe_read_failed", unit_id=unit_id, address=cursor
                )
                return None
            words.extend(int(register) for register in result.registers)
            cursor += chunk
            remaining -= chunk
        return words

    @staticmethod
    def _decode_candidates(
        cycle_ts: int, unit_id: int, start: int, words: list[int]
    ) -> list[tuple[int, int, int, str, float]]:
        """All plausible decodings: u16 per register, u32 BE/LE per pair."""
        rows: list[tuple[int, int, int, str, float]] = []

        def add(address: int, base: str, raw: int) -> None:
            for scale in _SCALES:
                decoding = base if scale == 1 else f"{base}/{scale}"
                rows.append((cycle_ts, unit_id, address, decoding, raw / scale))

        for offset, word in enumerate(words):
            add(start + offset, "u16", word)
        for offset in range(len(words) - 1):
            first, second = words[offset], words[offset + 1]
            add(start + offset, "u32be", (first << 16) | second)
            add(start + offset, "u32le", (second << 16) | first)
        return rows

    def _enqueue_summary(
        self, cycle_ts: int, rows: list[tuple[int, int, int, str, float]]
    ) -> None:
        """Enqueue ONE tier-3 summary for this cycle. Best-effort."""
        try:
            # Summarize scale-1 decodings only: /10 and /100 variants carry
            # the same monotonicity and would triple-fill the top-N slots.
            keys = sorted(
                {(r[1], r[2], r[3]) for r in rows if r[3] in _SCALE_ONE_DECODINGS}
            )
            candidates: list[dict[str, Any]] = []
            for unit_id, address, decoding in keys:
                series = self.db.get_register_candidate_series(
                    unit_id, address, decoding, limit=2
                )
                if not series:
                    continue
                last_value = float(series[0]["value"])
                delta: float | None = None
                if len(series) > 1:
                    delta = last_value - float(series[1]["value"])
                if delta is None:
                    hint = "unknown"  # first cycle — never fabricate a delta
                elif delta > 0:
                    hint = "increased"
                elif delta < 0:
                    hint = "decreased"
                else:
                    hint = "flat"
                candidates.append(
                    {
                        "unit": unit_id,
                        "address": address,
                        "decoding": decoding,
                        "last_value": last_value,
                        "delta_last_cycle": delta,
                        "monotonicity": hint,
                    }
                )
            candidates.sort(
                key=lambda c: (
                    c["delta_last_cycle"] is not None,
                    abs(c["delta_last_cycle"] or 0.0),
                ),
                reverse=True,
            )
            payload = {
                "type": "edge.register_probe",
                "title": "Register probe summary",
                "unit_ranges": self._ranges_spec,
                "cycle_at": datetime.fromtimestamp(cycle_ts / 1000, UTC).isoformat(),
                "candidates": candidates[:_SUMMARY_TOP_N],
            }
            # Priority 3 = backfill tier: ships only on healthy (>=4G) links,
            # so probe evidence never competes with alerts or telemetry.
            self.db.enqueue_sync(priority=3, payload=payload)
        except Exception:
            logger.exception("register_probe_summary_failed")


# ── Vendor-map validation mode (spec section 10) ─────────────────


class VendorMapProbe:
    """Read every register in a vendor map once and write a §10 record.

    This is the harness that turns "the map loads" into evidence about a
    PHYSICAL unit. It reads each enabled register exactly once through the
    production loader and the production decoder — deliberately, because a
    probe with its own decoding would validate the probe rather than the map.

    Read-only, opt-in, offline-safe. It never writes a Modbus register, never
    touches ``measurements``, persists its record to local SQLite first, and
    enqueues exactly one tier-3 summary that drains whenever a link appears.

    What it CANNOT establish, and says so in every record it writes:
    counter monotonicity (needs a second sweep hours later), sign conventions
    (needs the site's own meter), and the display comparison, wiring and
    operator name (need a human). Those stay as explicit blanks so that a
    record is visibly incomplete rather than quietly passing for a bench
    result.
    """

    def __init__(
        self,
        db: EdgeDatabase,
        vendor_map: str,
        transport: dict[str, Any],
        *,
        pace_seconds: float = 0.0,
    ) -> None:
        self.db = db
        self._map_name = vendor_map
        self._transport = dict(transport)
        self._pace_seconds = pace_seconds

    async def run_once(self, now_ms: int | None = None) -> dict[str, Any]:
        """Sweep the map once. Returns the record it stored.

        Raises VendorMapError for a map that cannot be trusted to poll (a
        template with blank addresses); every other failure degrades into a
        record that says what failed.
        """
        from collector.adapters.generic_modbus import GenericModbusAdapter
        from collector.adapters.modbus_decode import decode_register
        from collector.adapters.vendor_map import VendorMapError, load_vendor_map

        loaded = load_vendor_map(self._map_name)
        if not loaded.complete:
            raise VendorMapError(
                f"vendor map {self._map_name!r} is a template: fill the blank "
                f"addresses for {list(loaded.missing_addresses)} from this "
                "controller's own exported table before probing"
            )

        cycle_ts = now_ms if now_ms is not None else int(time.time() * 1000)
        adapter = GenericModbusAdapter()
        # The probe binds no assets: it is validating registers, not feeding
        # telemetry, and inventing asset UUIDs to satisfy the resolver would
        # put fabricated ids in an evidence record.
        connected = await adapter.initialize(
            {**self._transport, "vendor_map": self._map_name, "probe_mode": True}
        )
        # The adapter's own resolution is the one that matters: it is what a
        # poll would use. Re-loading the map here would sweep a DIFFERENT set
        # of registers from the one being validated.
        swept = adapter._map or loaded
        registers: list[dict[str, Any]] = []
        try:
            if connected:
                for index, spec in enumerate(swept.all_registers):
                    if spec.derived:
                        continue
                    if index and self._pace_seconds:
                        await asyncio.sleep(self._pace_seconds)
                    words = await adapter._read_words(spec, spec.word_count)
                    registers.append(
                        self._register_result(spec, words, decode_register)
                    )
        finally:
            await adapter.shutdown()

        record = self._build_record(swept, adapter, cycle_ts, connected, registers)
        self.db.insert_vendor_map_validation(
            timestamp=cycle_ts,
            map_name=swept.name,
            verification=swept.verification,
            record=record,
        )
        self._enqueue_summary(record)
        logger.info(
            "vendor_map_probe_complete",
            vendor_map=swept.name,
            connected=connected,
            registers=len(registers),
            read_ok=sum(1 for r in registers if r["status"] == "ok"),
        )
        return record

    @staticmethod
    def _register_result(
        spec: Any, words: list[int] | None, decode: Any
    ) -> dict[str, Any]:
        """One register's line in the record: raw words beside the decode."""
        result: dict[str, Any] = {
            "id": spec.id,
            "metric": spec.metric,
            "address": spec.address,
            "unit_id": spec.unit_id,
            "function_code": spec.function_code,
            "type": spec.data_type,
            "words": spec.word_count,
            "asset_role": spec.asset_role,
        }
        if words is None:
            # A read error is evidence too: it is how a technician finds a
            # wrong slave id, a missing option board, or an address this
            # firmware does not implement.
            result["status"] = "read_error"
            result["raw"] = None
            result["value"] = None
            return result

        decoded = decode(spec, words)
        result["raw"] = [f"0x{word:04X}" for word in words]
        if decoded.text is not None:
            result["status"] = "ok"
            result["value"] = decoded.text
        elif decoded.value is None:
            result["status"] = "no_value"
            result["value"] = None
            result["reason"] = decoded.refusal
        else:
            result["status"] = "ok"
            result["value"] = decoded.value
            result["quality"] = decoded.quality
            if decoded.quality:
                result["outside_declared_bounds"] = {
                    "min": spec.min_value,
                    "max": spec.max_value,
                }
        return result

    def _build_record(
        self,
        loaded: Any,
        adapter: Any,
        cycle_ts: int,
        connected: bool,
        registers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        redacted = {
            key: value
            for key, value in self._transport.items()
            if key
            in {
                "protocol",
                "host",
                "port",
                "baudrate",
                "parity",
                "stopbits",
                "bytesize",
                "unit_id",
            }
        }
        return {
            "schema": "gen318.vendor_map_validation.v1",
            "map": loaded.to_summary(),
            "verification": loaded.verification,
            # A record NEVER promotes a map. Promotion to bench or field is a
            # human decision made against a completed record, and the map file
            # is the only place the tier lives.
            "verification_note": (
                "This sweep is software evidence only. It does not promote the "
                "map above 'community'."
            ),
            "swept_at": datetime.fromtimestamp(cycle_ts / 1000, UTC).isoformat(),
            "connected": connected,
            "transport": redacted,
            "identity": adapter.identity() if hasattr(adapter, "identity") else {},
            "registers": registers,
            "totals": {
                "registers": len(registers),
                "ok": sum(1 for r in registers if r["status"] == "ok"),
                "no_value": sum(1 for r in registers if r["status"] == "no_value"),
                "read_error": sum(1 for r in registers if r["status"] == "read_error"),
                "suspect": sum(1 for r in registers if r.get("quality")),
            },
            # Everything below needs a second sweep, the site's own meter, or
            # a person. Present and empty, so the gaps are visible.
            "pending_checks": {
                "counter_monotonicity": (
                    "unknown — needs a later sweep; one reading cannot show a "
                    "counter increasing"
                ),
                "sign_conventions": (
                    "unconfirmed — compare battery and grid power against the "
                    "site's own meter under a known load"
                ),
                "display_comparison": (
                    "not done — compare at least five values against the "
                    "device's own display"
                ),
            },
            "operator": {
                "name": None,
                "wiring_port": None,
                "wiring_pins": None,
                "device_model": None,
                "device_serial": None,
                "device_firmware": None,
                "notes": None,
            },
        }

    def _enqueue_summary(self, record: dict[str, Any]) -> None:
        """One tier-3 item per sweep. Best-effort; never blocks the record."""
        try:
            self.db.enqueue_sync(
                priority=3,
                payload={
                    "type": "edge.vendor_map_validation",
                    "title": "Vendor map validation sweep",
                    "map": record["map"]["name"],
                    "vendor": record["map"]["vendor"],
                    "verification": record["verification"],
                    "swept_at": record["swept_at"],
                    "connected": record["connected"],
                    "identity": record["identity"],
                    "totals": record["totals"],
                    # The per-register detail stays local: it is large, and a
                    # backfill-tier item competing with telemetry on a 2G link
                    # is exactly what the tier ladder exists to prevent.
                    "detail": "stored locally in vendor_map_validations",
                },
            )
        except Exception:
            logger.exception("vendor_map_probe_summary_failed")


class SunSpecProbe:
    """Discovery sweep for a SunSpec device (spec section 5.2).

    SunSpec needs no register map, so its validation question is different:
    WHICH models does this unit present, how long is each, and what do the
    scale factors do to the raw words? The record answers that, and carries
    the same pending-checks and operator blanks as the vendor-map record —
    a discovery sweep is no more a bench result than a register sweep is.
    """

    def __init__(
        self,
        db: EdgeDatabase,
        transport: dict[str, Any],
        *,
        preset: str | None = None,
    ) -> None:
        self.db = db
        self._transport = dict(transport)
        self._preset = preset

    async def run_once(self, now_ms: int | None = None) -> dict[str, Any]:
        from collector.adapters.sunspec_modbus import SunSpecModbusAdapter

        cycle_ts = now_ms if now_ms is not None else int(time.time() * 1000)
        adapter = SunSpecModbusAdapter()
        connected = await adapter.initialize(self._transport)
        models: list[dict[str, Any]] = []
        readings: list[dict[str, Any]] = []
        try:
            if connected:
                for block in adapter._models:
                    entry: dict[str, Any] = {
                        "model_id": block.model_id,
                        "address": block.address,
                        "length": block.length,
                        "decoded_here": block.model_id in _sunspec_supported(),
                    }
                    try:
                        words = await adapter._read_words(
                            block.address, block.length + 2
                        )
                        entry["raw"] = [f"0x{word:04X}" for word in words]
                        entry["scale_factors"] = _sunspec_scale_factors(words)
                    except Exception:
                        entry["raw"] = None
                        entry["read_error"] = True
                    models.append(entry)
                for reading in await adapter.poll():
                    readings.append(
                        {
                            "metric": reading.metric_name,
                            "value": reading.value,
                            "quality": reading.quality,
                            "point": reading.raw_register,
                            "asset_id": reading.asset_id,
                        }
                    )
        finally:
            await adapter.shutdown()

        record = {
            "schema": "gen318.vendor_map_validation.v1",
            "map": {
                "name": self._preset or "sunspec",
                "vendor": (self._preset or "sunspec").replace("sunspec_", ""),
                "model_family": "SunSpec model chain",
                "source": (
                    "SunSpec Alliance model definitions, github.com/sunspec/models "
                    "@90b4a331; vendor preset connection facts in "
                    "docs/vendor-integrations/SOURCES.md"
                ),
                "verification": "community",
            },
            "verification": "community",
            "verification_note": (
                "Discovery evidence only. It does not promote anything above "
                "'community'; a SunSpec client is not a certification claim."
            ),
            "swept_at": datetime.fromtimestamp(cycle_ts / 1000, UTC).isoformat(),
            "connected": connected,
            "transport": {
                key: value
                for key, value in self._transport.items()
                if key in {"protocol", "host", "port", "unit_id", "base_addresses"}
            },
            "identity": {key: str(value) for key, value in adapter.nameplate().items()},
            "models": models,
            "registers": readings,
            "totals": {
                "models_discovered": len(models),
                "models_decoded": sum(1 for m in models if m["decoded_here"]),
                "readings": len(readings),
            },
            "pending_checks": {
                "counter_monotonicity": "unknown — needs a later sweep",
                "sign_conventions": (
                    "unconfirmed — compare the meter's signed power against the "
                    "site's own meter under a known load"
                ),
                "display_comparison": (
                    "not done — compare at least five values against the "
                    "device's own display"
                ),
            },
            "operator": {
                "name": None,
                "wiring_port": None,
                "device_model": None,
                "device_serial": None,
                "device_firmware": None,
                "model_type_selected": None,
                "notes": None,
            },
        }
        self.db.insert_vendor_map_validation(
            timestamp=cycle_ts,
            map_name=record["map"]["name"],
            verification="community",
            record=record,
        )
        try:
            self.db.enqueue_sync(
                priority=3,
                payload={
                    "type": "edge.sunspec_validation",
                    "title": "SunSpec discovery sweep",
                    "preset": record["map"]["name"],
                    "swept_at": record["swept_at"],
                    "connected": connected,
                    "models": [m["model_id"] for m in models],
                    "totals": record["totals"],
                    "identity": record["identity"],
                    "detail": "stored locally in vendor_map_validations",
                },
            )
        except Exception:
            logger.exception("sunspec_probe_summary_failed")
        logger.info(
            "sunspec_probe_complete",
            connected=connected,
            models=[m["model_id"] for m in models],
        )
        return record


def _sunspec_supported() -> set[int]:
    from collector.adapters.sunspec_modbus import _SUPPORTED_MODELS

    return set(_SUPPORTED_MODELS)


def _sunspec_scale_factors(words: list[int]) -> dict[str, int]:
    """Signed scale factors present in a model block.

    A sunssf is int16 with 0x8000 meaning "not implemented"; a block's
    plausible scale factors are the small signed values, which is what a
    technician needs to see beside the raw words to judge a decode.
    """
    factors: dict[str, int] = {}
    for offset, word in enumerate(words):
        if word == 0x8000:
            continue
        signed = word - 0x10000 if word & 0x8000 else word
        if -10 <= signed <= 10 and signed != 0:
            factors[f"offset_{offset}"] = signed
    return factors


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m collector.register_probe",
        description=(
            "Read-only Modbus probe. With --vendor-map, reads every register a "
            "shipped vendor map declares, once, and writes a validation record "
            "to the node's local database. Never writes to a device."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--vendor-map", help="a register map in vendor_maps/")
    mode.add_argument(
        "--sunspec",
        action="store_true",
        help="discover and dump a SunSpec model chain instead of reading a map",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="a sunspec_<vendor> preset to take connection defaults from",
    )
    parser.add_argument("--protocol", choices=("tcp", "rtu"), default=None)
    parser.add_argument("--host", default=None, help="Modbus TCP host")
    parser.add_argument(
        "--port",
        default=None,
        help="TCP port (default 502) or the serial device for RTU",
    )
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--parity", choices=("N", "E", "O"), default=None)
    parser.add_argument("--stopbits", type=int, choices=(1, 2), default=None)
    parser.add_argument("--bytesize", type=int, choices=(7, 8), default=None)
    parser.add_argument("--unit-id", type=int, default=None, help="Modbus slave id")
    parser.add_argument(
        "--request-spacing-ms",
        type=int,
        default=None,
        help="gap between requests (WiNet-S needs >= 20; ComAp answers one object at a time)",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("SQLITE_PATH", "/data/gen318.db"),
        help="edge SQLite path",
    )
    return parser


def _transport_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Only the options actually given; the map's own defaults fill the rest."""
    mapping = {
        "protocol": args.protocol,
        "host": args.host,
        "port": args.port,
        "baudrate": args.baudrate,
        "parity": args.parity,
        "stopbits": args.stopbits,
        "bytesize": args.bytesize,
        "unit_id": args.unit_id,
        "request_spacing_ms": args.request_spacing_ms,
    }
    transport = {key: value for key, value in mapping.items() if value is not None}
    if transport.get("protocol") == "tcp" and isinstance(transport.get("port"), str):
        transport["port"] = int(transport["port"])
    return transport


def _print_record(record: dict[str, Any]) -> None:
    """Human-readable sweep output — raw words beside every decode."""
    summary = record["map"]
    print(
        f"vendor map : {summary['name']}  ({summary['vendor']} {summary['model_family']})"
    )
    print(f"verification: {record['verification']}  — {record['verification_note']}")
    print(f"source      : {summary['source']}")
    print(f"swept at    : {record['swept_at']}")
    print(f"connected   : {record['connected']}")
    if record["identity"]:
        for key, value in record["identity"].items():
            print(f"identity    : {key} = {value}")
    print()
    header = (
        f"{'register':<22} {'addr':>6} {'type':<12} {'raw':<26} {'value':>14}  status"
    )
    print(header)
    print("-" * len(header))
    for entry in record["registers"]:
        raw = ",".join(entry["raw"]) if entry["raw"] else "-"
        value = entry["value"]
        shown = (
            "-"
            if value is None
            else (value if isinstance(value, str) else f"{value:.3f}")
        )
        status = entry["status"]
        if entry.get("reason"):
            status = f"{status} ({entry['reason']})"
        if entry.get("quality"):
            status += " OUTSIDE DECLARED BOUNDS"
        print(
            f"{entry['id']:<22} {entry['address']:>6} {entry['type']:<12} "
            f"{raw:<26} {shown:>14}  {status}"
        )
    totals = record["totals"]
    print()
    print(
        f"{totals['ok']} read, {totals['no_value']} no value, "
        f"{totals['read_error']} read errors, {totals['suspect']} outside bounds"
    )
    print()
    print("Still outstanding before this is a bench record:")
    for key, note in record["pending_checks"].items():
        print(f"  - {key}: {note}")


async def _run_cli(args: argparse.Namespace) -> int:
    db = EdgeDatabase(args.db, os.environ.get("SQLITE_ENCRYPTION_KEY", ""))
    db.connect()
    db.init_schema()
    try:
        transport = _transport_from_args(args)
        if args.sunspec:
            if args.preset:
                from collector.adapters.vendor_map import load_sunspec_preset

                preset = load_sunspec_preset(args.preset)
                # Explicit options still win over the preset's defaults.
                transport = {**preset.get("connection", {}), **transport}
            record = await SunSpecProbe(
                db=db, transport=transport, preset=args.preset
            ).run_once()
        else:
            record = await VendorMapProbe(
                db=db, vendor_map=args.vendor_map, transport=transport
            ).run_once()
    finally:
        db.close()
    _print_record(record)
    return 0 if record["connected"] else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        return asyncio.run(_run_cli(args))
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"probe failed: {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI surface
    raise SystemExit(main())
