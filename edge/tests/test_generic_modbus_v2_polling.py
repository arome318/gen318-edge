"""Block reads, per-group cadence and identity evidence (plan Task 4).

A Deye poll is ~60 registers across four assets. One request per register on
a 9600-baud RS485 line does not fit in a 30-second cadence, so registers ride
in blocks. The rules that keep that safe are what these tests pin: a block
never crosses a unit id or function code, never spans more than the house
limit, never swallows a large hole, and a block that faults falls back to
per-register reads so one unimplemented address cannot cost its neighbours.

Cadence is the other half: energy counters every 300 s and power every 30 s
is what makes the bus keep up, and a group that fell behind while the site
was offline is rescheduled from now rather than firing a catch-up burst.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collector.adapters import vendor_map as vm
from collector.adapters.generic_modbus import GenericModbusAdapter
from collector.adapters.vendor_map import RegisterSpec, resolve_vendor_map

INVERTER_ID = "30000000-0000-0000-0000-000000000001"
BATTERY_ID = "30000000-0000-0000-0000-000000000002"
SOLAR_ID = "30000000-0000-0000-0000-000000000003"


def _spec(**overrides: Any) -> RegisterSpec:
    base: dict[str, Any] = {
        "id": "r",
        "metric": "battery.voltage_v",
        "address": 100,
        "data_type": "uint16",
        "word_count": 1,
        "asset_role": "battery",
        "group": "power",
        "function_code": 3,
        "unit_id": 1,
    }
    base.update(overrides)
    return RegisterSpec(**base)


def _plan(
    specs: list[RegisterSpec], *, max_gap: int = 8, max_words: int = 120
) -> list[Any]:
    return GenericModbusAdapter._plan_blocks(
        specs, max_gap=max_gap, max_words=max_words
    )


# ── Block planning ──────────────────────────────────────────────


def test_contiguous_registers_become_one_request() -> None:
    specs = [_spec(id=f"r{a}", address=a) for a in (100, 101, 102, 103)]
    blocks = _plan(specs)
    assert len(blocks) == 1
    unit, fc, start, count, members = blocks[0]
    assert (unit, fc, start, count) == (1, 3, 100, 4)
    assert [m.id for m in members] == ["r100", "r101", "r102", "r103"]


def test_a_small_hole_is_read_through_rather_than_split() -> None:
    """Reading 3 wasted words beats paying for a second round trip."""
    specs = [_spec(id="a", address=100), _spec(id="b", address=104)]
    blocks = _plan(specs, max_gap=8)
    assert len(blocks) == 1
    assert blocks[0][2:4] == (100, 5)


def test_a_hole_larger_than_max_gap_splits_the_block() -> None:
    specs = [_spec(id="a", address=100), _spec(id="b", address=200)]
    blocks = _plan(specs, max_gap=8)
    assert len(blocks) == 2
    assert [b[2:4] for b in blocks] == [(100, 1), (200, 1)]


def test_max_gap_boundary_is_inclusive() -> None:
    """A gap of exactly max_gap still rides in one block; one more splits."""
    joined = _plan([_spec(id="a", address=100), _spec(id="b", address=109)], max_gap=8)
    split = _plan([_spec(id="a", address=100), _spec(id="b", address=110)], max_gap=8)
    assert len(joined) == 1
    assert len(split) == 2


def test_span_limit_counts_the_width_of_the_last_register() -> None:
    """A 2-word register at the edge must not push the request past the cap."""
    specs = [
        _spec(id="a", address=0),
        _spec(id="b", address=119, data_type="uint32", word_count=2),
    ]
    blocks = _plan(specs, max_gap=200, max_words=120)
    assert len(blocks) == 2  # 0..120 would be 121 words
    within = _plan(
        [
            _spec(id="a", address=0),
            _spec(id="b", address=118, data_type="uint32", word_count=2),
        ],
        max_gap=200,
        max_words=120,
    )
    assert len(within) == 1
    assert within[0][3] == 120


def test_blocks_never_cross_a_unit_id() -> None:
    specs = [
        _spec(id="a", address=100, unit_id=1),
        _spec(id="b", address=101, unit_id=2),
    ]
    blocks = _plan(specs)
    assert len(blocks) == 2
    assert {b[0] for b in blocks} == {1, 2}


def test_blocks_never_cross_a_function_code() -> None:
    """FC3 and FC4 are different address spaces; merging them reads garbage."""
    specs = [
        _spec(id="a", address=100, function_code=3),
        _spec(id="b", address=101, function_code=4),
    ]
    blocks = _plan(specs)
    assert len(blocks) == 2
    assert {b[1] for b in blocks} == {3, 4}


def test_overlapping_multiword_registers_share_one_block() -> None:
    """Deye reads 516-517 as a pair and 517 alone on some maps; both decode."""
    specs = [
        _spec(id="pair", address=516, data_type="uint32_le", word_count=2),
        _spec(id="high", address=517),
    ]
    blocks = _plan(specs)
    assert len(blocks) == 1
    assert blocks[0][2:4] == (516, 2)


def test_derived_registers_are_never_read() -> None:
    specs = [
        _spec(id="pv1", address=672),
        _spec(id="pv2", address=673),
        _spec(id="total", address=-1, derived="sum", components=("pv1", "pv2")),
    ]
    blocks = _plan(specs)
    assert len(blocks) == 1
    assert [m.id for m in blocks[0][4]] == ["pv1", "pv2"]


def test_unordered_input_is_planned_in_address_order() -> None:
    specs = [_spec(id=f"r{a}", address=a) for a in (103, 100, 102, 101)]
    blocks = _plan(specs)
    assert len(blocks) == 1
    assert blocks[0][2:4] == (100, 4)


def test_no_specs_plans_no_requests() -> None:
    assert _plan([]) == []


# ── Adapter-level: block reads, fallback, pacing ────────────────

BLOCK_MAP = textwrap.dedent(
    """
    vendor: testvendor
    model_family: blocks
    source: synthetic fixture (unit test)
    verification: community
    protocol_defaults:
      protocol: tcp
      unit_id: 1
      function_code: 3
      address_base: 0
      request_spacing_ms: 0
    assets: [inverter, battery, solar_array]
    groups:
      - name: power
        poll_interval_seconds: 30
        registers:
          - {id: inv_w, metric: inverter.ac_output_power_w, address: 100, type: int16,
             asset: inverter}
          - {id: bat_soc, metric: battery.soc_pct, address: 101, type: uint16,
             asset: battery}
          - {id: pv1, metric: solar.mppt1_power_w, address: 102, type: uint16,
             asset: solar_array}
          - {id: pv2, metric: solar.mppt2_power_w, address: 103, type: uint16,
             asset: solar_array}
          - {id: pv_total, metric: solar.pv_power_w, derived: sum, of: [pv1, pv2],
             asset: solar_array}
      - name: energy
        poll_interval_seconds: 300
        registers:
          - {id: pv_kwh, metric: solar.total_yield_kwh, address: 534,
             type: uint32_le, scale: 0.1, asset: solar_array}
    """
)


def _install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str = BLOCK_MAP
) -> None:
    (tmp_path / "block_map.yaml").write_text(text)
    monkeypatch.setattr(vm, "VENDOR_MAPS_DIR", tmp_path)


def _config(**extra: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "vendor_map": "block_map",
        "host": "192.168.1.10",
        "asset_mappings": {
            "inverter": INVERTER_ID,
            "battery": BATTERY_ID,
            "solar_array": SOLAR_ID,
        },
    }
    config.update(extra)
    return config


def _tcp_client(words_by_address: dict[int, list[int]]) -> AsyncMock:
    flat: dict[int, int] = {}
    for address, registers in words_by_address.items():
        for offset, word in enumerate(registers):
            flat[address + offset] = word

    client = AsyncMock()
    client.connect = AsyncMock(return_value=True)
    client.connected = True
    client.close = MagicMock()

    # pymodbus renamed slave -> device_id; the adapter passes whichever the
    # installed version accepts, so the fake accepts both.
    async def _read(address: int, count: int, **unit: int) -> MagicMock:
        result = MagicMock()
        try:
            registers = [flat[address + offset] for offset in range(count)]
        except KeyError:
            result.isError.return_value = True
            return result
        result.isError.return_value = False
        result.registers = registers
        return result

    client.read_holding_registers = AsyncMock(side_effect=_read)
    client.read_input_registers = AsyncMock(side_effect=_read)
    return client


ALL_WORDS = {100: [2500], 101: [88], 102: [600], 103: [700], 534: [0x0002, 0x0001]}


async def _start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, words: dict[str, Any], **extra: Any
) -> tuple[GenericModbusAdapter, AsyncMock]:
    _install(tmp_path, monkeypatch, extra.pop("map_text", BLOCK_MAP))
    adapter = GenericModbusAdapter()
    client = _tcp_client(words)
    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        MockClient.return_value = client
        assert await adapter.initialize(_config(**extra)) is True
    return adapter, client


@pytest.mark.asyncio
async def test_a_poll_reads_contiguous_registers_in_one_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, client = await _start(tmp_path, monkeypatch, ALL_WORDS)
    readings = await adapter.poll()

    calls = [
        (c.kwargs["address"], c.kwargs["count"])
        for c in client.read_holding_registers.call_args_list
    ]
    assert (100, 4) in calls  # the four power registers, one request
    assert (534, 2) in calls  # the energy counter, its own block
    assert len(calls) == 2
    values = {r.metric_name: r.value for r in readings}
    assert values["inverter.ac_output_power_w"] == 2500.0
    assert values["battery.soc_pct"] == 88.0
    assert values["solar.pv_power_w"] == 1300.0  # derived from the block slice
    assert values["solar.total_yield_kwh"] == pytest.approx(6553.8)


@pytest.mark.asyncio
async def test_each_register_decodes_from_its_own_slice_of_the_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An off-by-one in the slice silently swaps two assets' readings."""
    adapter, _ = await _start(tmp_path, monkeypatch, ALL_WORDS)
    readings = await adapter.poll()
    by_metric = {r.metric_name: r for r in readings}

    assert by_metric["inverter.ac_output_power_w"].value == 2500.0
    assert by_metric["inverter.ac_output_power_w"].asset_id == INVERTER_ID
    assert by_metric["battery.soc_pct"].value == 88.0
    assert by_metric["battery.soc_pct"].asset_id == BATTERY_ID
    assert by_metric["solar.mppt1_power_w"].value == 600.0
    assert by_metric["solar.mppt2_power_w"].value == 700.0


@pytest.mark.asyncio
async def test_a_faulting_block_falls_back_to_per_register_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deye faults on some reserved addresses inside an otherwise good span."""
    words = dict(ALL_WORDS)
    del words[102]  # unimplemented in the middle of the power block
    adapter, client = await _start(tmp_path, monkeypatch, words)
    readings = await adapter.poll()

    values = {r.metric_name: r.value for r in readings}
    assert values["inverter.ac_output_power_w"] == 2500.0
    assert values["battery.soc_pct"] == 88.0
    assert values["solar.mppt2_power_w"] == 700.0
    assert "solar.mppt1_power_w" not in values
    # The derived sum is withheld: one component never read.
    assert "solar.pv_power_w" not in values
    # One failed block read, then one request per register in that block.
    singles = [
        c
        for c in client.read_holding_registers.call_args_list
        if c.kwargs["count"] == 1
    ]
    assert len(singles) == 4


@pytest.mark.asyncio
async def test_a_short_block_response_also_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(tmp_path, monkeypatch)
    adapter = GenericModbusAdapter()
    client = _tcp_client(ALL_WORDS)
    real_read = client.read_holding_registers.side_effect

    async def _short(address: int, count: int, **unit: int) -> MagicMock:
        result = await real_read(address=address, count=count, **unit)
        if count > 1 and not result.isError():
            result.registers = result.registers[:-1]  # a truncated response
        return result

    client.read_holding_registers = AsyncMock(side_effect=_short)
    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        MockClient.return_value = client
        await adapter.initialize(_config())
        readings = await adapter.poll()

    values = {r.metric_name: r.value for r in readings}
    assert values["inverter.ac_output_power_w"] == 2500.0
    assert values["battery.soc_pct"] == 88.0


@pytest.mark.asyncio
async def test_a_block_read_exception_does_not_escape_the_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline-first: a transport error is a quiet empty poll, never a crash."""
    _install(tmp_path, monkeypatch)
    adapter = GenericModbusAdapter()
    client = _tcp_client(ALL_WORDS)
    client.read_holding_registers = AsyncMock(side_effect=OSError("bus fell over"))
    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        MockClient.return_value = client
        await adapter.initialize(_config())
        readings = await adapter.poll()

    assert readings == []


@pytest.mark.asyncio
async def test_request_spacing_is_applied_between_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WiNet-S needs >= 20 ms; a ComAp answers one object at a time."""
    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    adapter, _ = await _start(tmp_path, monkeypatch, ALL_WORDS, request_spacing_ms=20)
    monkeypatch.setattr("collector.adapters.generic_modbus.asyncio.sleep", _fake_sleep)
    await adapter.poll()

    # Two blocks -> one gap between them.
    assert slept == [pytest.approx(0.020)]


@pytest.mark.asyncio
async def test_no_spacing_configured_means_no_sleeping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    adapter, _ = await _start(tmp_path, monkeypatch, ALL_WORDS)
    monkeypatch.setattr("collector.adapters.generic_modbus.asyncio.sleep", _fake_sleep)
    await adapter.poll()
    assert slept == []


@pytest.mark.asyncio
async def test_poll_while_disconnected_reads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, client = await _start(tmp_path, monkeypatch, ALL_WORDS)
    client.connected = False
    assert await adapter.poll() == []
    assert client.read_holding_registers.call_count == 0


# ── Per-group cadence on a fake clock ───────────────────────────


@pytest.mark.asyncio
async def test_the_first_poll_reads_every_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _ = await _start(tmp_path, monkeypatch, ALL_WORDS)
    metrics = {r.metric_name for r in await adapter.poll()}
    assert "inverter.ac_output_power_w" in metrics
    assert "solar.total_yield_kwh" in metrics


@pytest.mark.asyncio
async def test_a_slow_group_is_skipped_until_its_cadence_comes_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """30 s power, 300 s energy — the point of per-group timers."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        "collector.adapters.generic_modbus.time.monotonic", lambda: clock["now"]
    )
    adapter, client = await _start(tmp_path, monkeypatch, ALL_WORDS)

    await adapter.poll()  # both groups
    client.read_holding_registers.reset_mock()

    clock["now"] += 30.0
    metrics = {r.metric_name for r in await adapter.poll()}
    assert "inverter.ac_output_power_w" in metrics
    assert "solar.total_yield_kwh" not in metrics
    assert [
        c.kwargs["address"] for c in client.read_holding_registers.call_args_list
    ] == [100]

    clock["now"] += 270.0  # 300 s since the first poll
    metrics = {r.metric_name for r in await adapter.poll()}
    assert "solar.total_yield_kwh" in metrics


@pytest.mark.asyncio
async def test_a_group_polled_early_reads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        "collector.adapters.generic_modbus.time.monotonic", lambda: clock["now"]
    )
    adapter, client = await _start(tmp_path, monkeypatch, ALL_WORDS)
    await adapter.poll()
    client.read_holding_registers.reset_mock()

    clock["now"] += 5.0
    assert await adapter.poll() == []
    assert client.read_holding_registers.call_count == 0


@pytest.mark.asyncio
async def test_a_long_outage_does_not_produce_a_catch_up_burst(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After an hour away the device gets one poll, not 120 queued ones."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        "collector.adapters.generic_modbus.time.monotonic", lambda: clock["now"]
    )
    adapter, client = await _start(tmp_path, monkeypatch, ALL_WORDS)
    await adapter.poll()

    clock["now"] += 3600.0
    client.read_holding_registers.reset_mock()
    await adapter.poll()
    first_catch_up = client.read_holding_registers.call_count

    client.read_holding_registers.reset_mock()
    await adapter.poll()  # same instant, nothing is due again
    assert client.read_holding_registers.call_count == 0
    assert first_catch_up == 2  # one block per group, once

    clock["now"] += 30.0
    client.read_holding_registers.reset_mock()
    await adapter.poll()
    assert [
        c.kwargs["address"] for c in client.read_holding_registers.call_args_list
    ] == [100]


@pytest.mark.asyncio
async def test_the_monotonic_clock_is_what_schedules_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wall-clock steps (NTP finding the network) must not fire every group."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        "collector.adapters.generic_modbus.time.monotonic", lambda: clock["now"]
    )
    adapter, client = await _start(tmp_path, monkeypatch, ALL_WORDS)
    await adapter.poll()
    client.read_holding_registers.reset_mock()

    # A large wall-clock jump with the monotonic clock unmoved.
    monkeypatch.setattr("collector.adapters.generic_modbus.time.time", lambda: 9e9)
    assert await adapter.poll() == []
    assert client.read_holding_registers.call_count == 0


# ── Identity evidence reaches the sync queue as tier 3 ──────────

# BLOCK_MAP is already dedented, so the top-level keys sit at column 0.
IDENTITY_MAP = BLOCK_MAP.replace(
    "assets: [inverter, battery, solar_array]",
    "identity:\n"
    "  serial: {address: 3, type: string, words: 2}\n"
    "assets: [inverter, battery, solar_array]",
)
assert "identity:" in IDENTITY_MAP


def _collector(edge_config: Any, edge_db: Any) -> Any:
    from collector.main import CollectorService

    service = CollectorService(edge_config)
    service.db = edge_db
    return service


@pytest.mark.asyncio
async def test_identity_evidence_is_queued_once_at_tier_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_config: Any, edge_db: Any
) -> None:
    words = dict(ALL_WORDS)
    words[3] = [0x4142, 0x4344]  # "ABCD"
    adapter, _ = await _start(tmp_path, monkeypatch, words, map_text=IDENTITY_MAP)

    service = _collector(edge_config, edge_db)
    service._queue_vendor_map_evidence("deye-hybrid", adapter)

    items = edge_db.get_queue_items(priority=3, limit=10)
    assert len(items) == 1
    payload = items[0]["payload"]
    assert payload["type"] == "edge.vendor_map_identity"
    assert payload["adapter"] == "deye-hybrid"
    assert payload["vendor"] == "testvendor"
    assert payload["verification"] == "community"
    assert payload["identity"] == {"serial": "ABCD"}
    assert payload["source"]
    # Nothing at a higher tier: evidence never competes with alerts.
    assert edge_db.get_queue_items(priority=0, limit=10) == []
    assert edge_db.get_queue_items(priority=2, limit=10) == []


@pytest.mark.asyncio
async def test_identity_evidence_records_the_map_even_when_the_device_is_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_config: Any, edge_db: Any
) -> None:
    """An unreadable serial register still leaves a record of which map ran."""
    adapter, _ = await _start(tmp_path, monkeypatch, ALL_WORDS, map_text=IDENTITY_MAP)

    service = _collector(edge_config, edge_db)
    service._queue_vendor_map_evidence("deye-hybrid", adapter)

    payload = edge_db.get_queue_items(priority=3, limit=10)[0]["payload"]
    assert payload["identity"] == {}
    assert payload["name"] == "block_map"
    assert payload["verification"] == "community"


def test_a_legacy_adapter_queues_no_vendor_map_evidence(
    edge_config: Any, edge_db: Any
) -> None:
    service = _collector(edge_config, edge_db)
    service._queue_vendor_map_evidence("victron", object())
    assert edge_db.get_queue_items(priority=3, limit=10) == []


def test_an_evidence_failure_never_propagates(edge_config: Any, edge_db: Any) -> None:
    """Edge invariant 1: evidence is best-effort, adapter startup is not."""

    class Exploding:
        def vendor_map_summary(self) -> dict[str, Any]:
            raise RuntimeError("boom")

    service = _collector(edge_config, edge_db)
    service._queue_vendor_map_evidence("broken", Exploding())  # must not raise
    assert edge_db.get_queue_items(priority=3, limit=10) == []


@pytest.mark.asyncio
async def test_identity_is_read_once_not_on_every_poll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    words = dict(ALL_WORDS)
    words[3] = [0x4142, 0x4344]
    adapter, client = await _start(tmp_path, monkeypatch, words, map_text=IDENTITY_MAP)
    after_init = [
        c.kwargs["address"] for c in client.read_holding_registers.call_args_list
    ]

    await adapter.poll()
    await adapter.poll()
    all_calls = [
        c.kwargs["address"] for c in client.read_holding_registers.call_args_list
    ]

    assert after_init == [3]
    assert all_calls.count(3) == 1


@pytest.mark.asyncio
async def test_a_reconnect_rereads_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Firmware can change between sessions; the record must follow the unit."""
    words = dict(ALL_WORDS)
    words[3] = [0x4142, 0x4344]
    adapter, client = await _start(tmp_path, monkeypatch, words, map_text=IDENTITY_MAP)
    assert adapter.identity() == {"serial": "ABCD"}

    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        new_words = dict(words)
        new_words[3] = [0x5859, 0x5A5A]  # "XYZZ"
        MockClient.return_value = _tcp_client(new_words)
        await adapter.initialize(_config())

    assert adapter.identity() == {"serial": "XYZZ"}


# ── The map path still cannot write ─────────────────────────────


@pytest.mark.asyncio
async def test_block_polling_never_calls_a_write_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, client = await _start(tmp_path, monkeypatch, ALL_WORDS)
    await adapter.poll()
    client.write_register.assert_not_called()
    client.write_registers.assert_not_called()
    assert adapter.capabilities().can_write is False


def test_resolved_map_exposes_no_write_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(tmp_path, monkeypatch)
    resolved = resolve_vendor_map(
        "block_map",
        {
            "asset_mappings": {
                "inverter": INVERTER_ID,
                "battery": BATTERY_ID,
                "solar_array": SOLAR_ID,
            }
        },
    )
    assert not hasattr(resolved, "write_mappings")
    assert all(spec.function_code in (3, 4) for spec in resolved.all_registers)
