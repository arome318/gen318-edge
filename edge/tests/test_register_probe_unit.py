"""Tests for the opt-in Modbus register probe (battery lifetime counters).

The probe is mapping-decision evidence gathering: it records candidate
decodings (u16, u32 BE, u32 LE at scales 1/10/100) into SQLite and ships
one tier-3 summary per cycle. It must never fail or block when pymodbus
or the device is unavailable.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collector.register_probe import (
    RegisterProbe,
    parse_ranges,
)

# ── Helpers ─────────────────────────────────────────────────────


def _make_mock_client(register_values: dict[int, dict[int, int]]) -> AsyncMock:
    """Mock AsyncModbusTcpClient serving registers from {unit: {addr: raw}}."""
    client = AsyncMock()
    client.connect = AsyncMock(return_value=True)
    client.close = MagicMock()

    # pymodbus calls the unit id `slave` before 3.11 and `device_id` from
    # 3.14; the probe passes whichever the installed version accepts, so the
    # fake takes either rather than pinning one.
    async def _read(address: int, count: int, **unit: int) -> MagicMock:
        result = MagicMock()
        unit_regs = register_values.get(unit.get("device_id", unit.get("slave")))
        if unit_regs is None:
            result.isError.return_value = True
            return result
        try:
            registers = [unit_regs[a] for a in range(address, address + count)]
        except KeyError:
            result.isError.return_value = True
            return result
        result.isError.return_value = False
        result.registers = registers
        return result

    client.read_holding_registers = AsyncMock(side_effect=_read)
    return client


def _make_probe(edge_db: Any, ranges: str = "225:284-286") -> RegisterProbe:
    return RegisterProbe(
        db=edge_db,
        host="192.168.1.1",
        port=502,
        ranges=ranges,
        interval_seconds=3600,
        pace_seconds=0.0,
    )


# ── Range parsing ───────────────────────────────────────────────


def test_parse_ranges_valid() -> None:
    assert parse_ranges("225:280-310,225:325-335") == [
        (225, 280, 310),
        (225, 325, 335),
    ]


def test_parse_ranges_single_register() -> None:
    """A bare 'unit:addr' entry probes exactly one register."""
    assert parse_ranges("247:790") == [(247, 790, 790)]


def test_parse_ranges_invalid_entries_skipped() -> None:
    """Garbage, reversed, out-of-range, and oversized entries are skipped."""
    assert parse_ranges("garbage") == []
    assert parse_ranges("225:10-5") == []  # reversed
    assert parse_ranges("225:65530-65600") == []  # beyond register space
    assert parse_ranges("225:0-9999") == []  # oversized sweep
    assert parse_ranges("") == []
    # Valid entries survive alongside invalid ones
    assert parse_ranges("bad,225:284-286,also:bad") == [(225, 284, 286)]


# ── Candidate decoding ──────────────────────────────────────────


def test_decode_candidates_u32_be_vs_le() -> None:
    """Adjacent pairs decode both word orders: BE = lower address is MSW."""
    words = [0x0001, 0x0002]  # [hi, lo] in Victron (BE) convention
    rows = RegisterProbe._decode_candidates(1000, 225, 300, words)
    by_key = {(r[2], r[3]): r[4] for r in rows}

    assert by_key[(300, "u16")] == pytest.approx(1.0)
    assert by_key[(301, "u16")] == pytest.approx(2.0)
    assert by_key[(300, "u32be")] == pytest.approx((0x0001 << 16) | 0x0002)  # 65538
    assert by_key[(300, "u32le")] == pytest.approx((0x0002 << 16) | 0x0001)  # 131073

    # Reversed word order flips the two u32 decodings
    rows_rev = RegisterProbe._decode_candidates(1000, 225, 300, [0x0002, 0x0001])
    by_key_rev = {(r[2], r[3]): r[4] for r in rows_rev}
    assert by_key_rev[(300, "u32be")] == pytest.approx((0x0002 << 16) | 0x0001)
    assert by_key_rev[(300, "u32le")] == pytest.approx((0x0001 << 16) | 0x0002)


def test_decode_candidates_scales() -> None:
    """Every decoding is recorded at scales 1, 10 and 100 (divide)."""
    rows = RegisterProbe._decode_candidates(1000, 225, 301, [65535])
    by_key = {(r[2], r[3]): r[4] for r in rows}

    assert by_key[(301, "u16")] == pytest.approx(65535.0)
    assert by_key[(301, "u16/10")] == pytest.approx(6553.5)  # the kWh cap value
    assert by_key[(301, "u16/100")] == pytest.approx(655.35)
    # Single register → no u32 pair candidates
    assert not any(r[3].startswith("u32") for r in rows)
    # Row shape matches insert_register_candidates
    assert rows[0][0] == 1000  # timestamp
    assert rows[0][1] == 225  # unit_id


# ── SQLite round-trip ───────────────────────────────────────────


def test_register_candidates_insert_and_prune(edge_db: Any) -> None:
    batch = [
        (1000, 225, 284, "u16", 100.0),
        (1000, 225, 284, "u32be", 6553600.0),
        (2000, 225, 284, "u16", 101.0),
    ]
    assert edge_db.insert_register_candidates(batch) == 3
    # Duplicate insert is ignored (INSERT OR IGNORE on the PK)
    edge_db.insert_register_candidates([(1000, 225, 284, "u16", 999.0)])

    series = edge_db.get_register_candidate_series(225, 284, "u16", limit=10)
    assert [row["timestamp"] for row in series] == [2000, 1000]  # newest first
    assert series[1]["value"] == pytest.approx(100.0)  # duplicate did not clobber

    pruned = edge_db.prune_register_candidates(older_than_ms=1500)
    assert pruned == 2  # both ts=1000 rows (u16 + u32be)
    series = edge_db.get_register_candidate_series(225, 284, "u16", limit=10)
    assert [row["timestamp"] for row in series] == [2000]


# ── Full cycle: persist + summary ───────────────────────────────


@pytest.mark.asyncio
async def test_run_cycle_persists_and_enqueues_summary(edge_db: Any) -> None:
    probe = _make_probe(edge_db, ranges="225:284-286")
    values = {225: {284: 100, 285: 0, 286: 5}}
    mock_client = _make_mock_client(values)
    probe._client_cls = MagicMock(return_value=mock_client)

    await probe._run_cycle(now_ms=1000)
    values[225][284] = 101  # counter ticked between cycles
    await probe._run_cycle(now_ms=2000)

    # Raw u16 series persisted for each register in the range
    series = edge_db.get_register_candidate_series(225, 284, "u16", limit=10)
    assert [row["value"] for row in series] == [101.0, 100.0]

    # Exactly one tier-3 summary per cycle
    items = edge_db.get_queue_items(priority=3, limit=10)
    assert len(items) == 2
    payload = items[-1]["payload"]
    assert payload["type"] == "edge.register_probe"
    assert payload["title"] == "Register probe summary"
    assert payload["unit_ranges"] == "225:284-286"
    assert "cycle_at" in payload

    candidates = payload["candidates"]
    assert 0 < len(candidates) <= 20
    by_key = {(c["unit"], c["address"], c["decoding"]): c for c in candidates}

    ticked = by_key[(225, 284, "u16")]
    assert ticked["last_value"] == pytest.approx(101.0)
    assert ticked["delta_last_cycle"] == pytest.approx(1.0)
    assert ticked["monotonicity"] == "increased"

    flat = by_key[(225, 285, "u16")]
    assert flat["delta_last_cycle"] == pytest.approx(0.0)
    assert flat["monotonicity"] == "flat"

    # Largest absolute delta ranks first: u32be at 284 moved by 65536
    assert candidates[0]["decoding"] == "u32be"
    assert candidates[0]["address"] == 284
    assert candidates[0]["delta_last_cycle"] == pytest.approx(65536.0)

    # First cycle had no history — deltas unknown, never fabricated
    first_payload = items[0]["payload"]
    first = {
        (c["unit"], c["address"], c["decoding"]): c for c in first_payload["candidates"]
    }
    assert first[(225, 284, "u16")]["delta_last_cycle"] is None
    assert first[(225, 284, "u16")]["monotonicity"] == "unknown"


@pytest.mark.asyncio
async def test_run_cycle_prunes_old_candidates(edge_db: Any) -> None:
    probe = _make_probe(edge_db)
    probe._client_cls = MagicMock(
        return_value=_make_mock_client({225: {a: 1 for a in range(284, 287)}})
    )
    # A candidate row far older than the 7-day retention window
    stale_ts = 1  # epoch ms — ancient
    edge_db.insert_register_candidates([(stale_ts, 225, 284, "u16", 42.0)])

    await probe._run_cycle()

    series = edge_db.get_register_candidate_series(225, 284, "u16", limit=100)
    assert all(row["timestamp"] != stale_ts for row in series)


# ── Degradation: never fail, never block ────────────────────────


@pytest.mark.asyncio
async def test_pymodbus_missing_graceful_noop(edge_db: Any) -> None:
    """Probe constructed without pymodbus degrades to a logged no-op."""
    with patch.dict(sys.modules, {"pymodbus": None, "pymodbus.client": None}):
        probe = _make_probe(edge_db)
    assert probe._client_cls is None
    # Must not raise, must not enqueue anything
    await probe._run_cycle(now_ms=1000)
    assert edge_db.get_queue_items(priority=3, limit=10) == []


@pytest.mark.asyncio
async def test_connect_failure_graceful(edge_db: Any) -> None:
    probe = _make_probe(edge_db)
    mock_client = _make_mock_client({})
    mock_client.connect = AsyncMock(return_value=False)
    probe._client_cls = MagicMock(return_value=mock_client)

    await probe._run_cycle(now_ms=1000)  # must not raise

    assert edge_db.get_queue_items(priority=3, limit=10) == []
    assert edge_db.get_register_candidate_series(225, 284, "u16", limit=10) == []


@pytest.mark.asyncio
async def test_read_error_skips_range_without_raising(edge_db: Any) -> None:
    """A unit that errors on read produces no rows but doesn't kill the cycle."""
    probe = _make_probe(edge_db, ranges="225:284-286,247:790-791")
    # Only unit 247 answers; 225 errors
    mock_client = _make_mock_client({247: {790: 7, 791: 8}})
    probe._client_cls = MagicMock(return_value=mock_client)

    await probe._run_cycle(now_ms=1000)

    assert edge_db.get_register_candidate_series(225, 284, "u16", limit=10) == []
    series = edge_db.get_register_candidate_series(247, 790, "u32be", limit=10)
    assert len(series) == 1
    assert series[0]["value"] == pytest.approx((7 << 16) | 8)


# ── Collector wiring: flag off = no task ────────────────────────


def _make_collector_service(edge_config: Any, **overrides: Any) -> Any:
    from collector.main import CollectorService

    config = dataclasses.replace(edge_config, **overrides)
    return CollectorService(config)


def test_flag_off_registers_no_task(edge_config: Any) -> None:
    service = _make_collector_service(edge_config)  # default: disabled
    service._register_probe_task()
    assert "register-probe" not in service._supervisor._factories


def test_flag_on_without_host_registers_no_task(edge_config: Any) -> None:
    service = _make_collector_service(
        edge_config, register_probe_enabled=True, register_probe_host=""
    )
    service._register_probe_task()
    assert "register-probe" not in service._supervisor._factories


def test_flag_on_with_host_registers_task(edge_config: Any) -> None:
    service = _make_collector_service(
        edge_config,
        register_probe_enabled=True,
        register_probe_host="192.168.1.9",
    )
    service._register_probe_task()
    assert "register-probe" in service._supervisor._factories


# ── Config parsing ──────────────────────────────────────────────


def test_config_defaults_and_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    from common.config import load_config

    # Defaults: disabled, no host, documented ranges
    for var in (
        "EDGE_REGISTER_PROBE_ENABLED",
        "EDGE_REGISTER_PROBE_HOST",
        "EDGE_REGISTER_PROBE_PORT",
        "EDGE_REGISTER_PROBE_RANGES",
        "EDGE_REGISTER_PROBE_INTERVAL_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)
    config = load_config()
    assert config.register_probe_enabled is False
    assert config.register_probe_host == ""
    assert config.register_probe_port == 502
    assert config.register_probe_ranges == "225:280-310,225:325-335"
    assert config.register_probe_interval_seconds == 3600

    monkeypatch.setenv("EDGE_REGISTER_PROBE_ENABLED", "true")
    monkeypatch.setenv("EDGE_REGISTER_PROBE_HOST", "192.168.1.9")
    monkeypatch.setenv("EDGE_REGISTER_PROBE_PORT", "1502")
    monkeypatch.setenv("EDGE_REGISTER_PROBE_RANGES", "247:790-791")
    monkeypatch.setenv("EDGE_REGISTER_PROBE_INTERVAL_SECONDS", "600")
    config = load_config()
    assert config.register_probe_enabled is True
    assert config.register_probe_host == "192.168.1.9"
    assert config.register_probe_port == 1502
    assert config.register_probe_ranges == "247:790-791"
    assert config.register_probe_interval_seconds == 600
