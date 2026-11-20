"""Vendor-map validation probe and its §10 record (plan Task 6).

The probe is what turns "the map loads" into evidence about a physical
unit. It reads every register a map declares, once, through the production
loader and the production decoder — a probe with its own decoding would
validate the probe rather than the map.

The record it writes has to be honest about what a single sweep cannot
establish. Counter monotonicity needs a second sweep, sign conventions need
the site's own meter, and the wiring, display comparison and operator name
need a person. Those stay as visible blanks, so an incomplete record cannot
quietly pass for a bench result.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collector.adapters import vendor_map as vm
from collector.adapters.vendor_map import VendorMapError
from collector.register_probe import (
    VendorMapProbe,
    _build_arg_parser,
    _transport_from_args,
)

PROBE_MAP = textwrap.dedent(
    """
    vendor: testvendor
    model_family: probe
    source: synthetic fixture (unit test) @ rev 0
    verification: community
    protocol_defaults:
      protocol: tcp
      port: 502
      unit_id: 1
      function_code: 3
      address_base: 0
    identity:
      serial: {address: 3, type: string, words: 2}
    assets: [inverter, battery]
    optional_assets: [diesel_genset]
    features:
      gen_port: false
    groups:
      - name: power
        poll_interval_seconds: 30
        registers:
          - {id: inv_w, metric: inverter.ac_output_power_w, address: 100,
             type: int16, asset: inverter}
          - {id: bat_soc, metric: battery.soc_pct, address: 101, type: uint16,
             asset: battery, min: 0, max: 100}
          - {id: inv_state, metric: inverter.state, address: 102, type: enum,
             values: {0: 0, 2: 2}, asset: inverter}
          - {id: missing, metric: battery.temperature_c, address: 900,
             type: int16, scale: 0.1, asset: battery}
          - {id: gen_w, metric: genset.power_w, address: 103, type: int16,
             asset: diesel_genset, feature: gen_port}
    """
)

TEMPLATE_MAP = PROBE_MAP.replace("address: 100", "address: null")


def _install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str = PROBE_MAP
) -> None:
    (tmp_path / "probe_map.yaml").write_text(text)
    monkeypatch.setattr(vm, "VENDOR_MAPS_DIR", tmp_path)


def _client(words_by_address: dict[int, list[int]]) -> AsyncMock:
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


# "AB" at 3-4, 2500 W, 88 %, state 2; 900 is unimplemented.
GOOD_WORDS = {3: [0x4142, 0x4344], 100: [2500], 101: [88], 102: [2], 103: [400]}


async def _sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edge_db: Any,
    words: dict[int, list[int]] | None = None,
    *,
    text: str = PROBE_MAP,
    connect: bool = True,
) -> dict[str, Any]:
    _install(tmp_path, monkeypatch, text)
    client = _client(GOOD_WORDS if words is None else words)
    client.connect = AsyncMock(return_value=connect)
    client.connected = connect
    probe = VendorMapProbe(
        db=edge_db,
        vendor_map="probe_map",
        transport={"protocol": "tcp", "host": "192.168.1.10"},
    )
    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        MockClient.return_value = client
        return await probe.run_once(now_ms=1_700_000_000_000)


# ── The sweep reads what the map declares ───────────────────────


@pytest.mark.asyncio
async def test_a_sweep_reads_every_declared_register_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    record = await _sweep(tmp_path, monkeypatch, edge_db)
    by_id = {entry["id"]: entry for entry in record["registers"]}

    assert by_id["inv_w"]["value"] == 2500.0
    assert by_id["bat_soc"]["value"] == 88.0
    assert by_id["inv_state"]["value"] == 2.0
    assert record["totals"]["registers"] == 5
    assert record["totals"]["ok"] == 4


@pytest.mark.asyncio
async def test_a_feature_gated_register_is_still_probed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    """The point of a sweep is to learn what the device actually answers."""
    record = await _sweep(tmp_path, monkeypatch, edge_db)
    by_id = {entry["id"]: entry for entry in record["registers"]}
    assert by_id["gen_w"]["value"] == 400.0


@pytest.mark.asyncio
async def test_raw_words_sit_beside_every_decoded_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    """The raw words are the evidence; the decode is this map's reading of it."""
    record = await _sweep(tmp_path, monkeypatch, edge_db)
    by_id = {entry["id"]: entry for entry in record["registers"]}
    assert by_id["inv_w"]["raw"] == ["0x09C4"]
    assert by_id["bat_soc"]["raw"] == ["0x0058"]


@pytest.mark.asyncio
async def test_a_read_error_is_recorded_not_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    """A read error finds a wrong slave id or an unimplemented address."""
    record = await _sweep(tmp_path, monkeypatch, edge_db)
    by_id = {entry["id"]: entry for entry in record["registers"]}

    assert by_id["missing"]["status"] == "read_error"
    assert by_id["missing"]["value"] is None
    assert by_id["missing"]["raw"] is None
    assert record["totals"]["read_error"] == 1


@pytest.mark.asyncio
async def test_an_unmapped_enum_code_is_reported_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    words = dict(GOOD_WORDS)
    words[102] = [9]  # not in the map's table
    record = await _sweep(tmp_path, monkeypatch, edge_db, words)
    by_id = {entry["id"]: entry for entry in record["registers"]}

    assert by_id["inv_state"]["status"] == "no_value"
    assert by_id["inv_state"]["reason"] == "unknown_enum"
    assert by_id["inv_state"]["raw"] == ["0x0009"]
    assert by_id["inv_state"]["value"] is None


@pytest.mark.asyncio
async def test_a_value_outside_declared_bounds_is_flagged_but_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    words = dict(GOOD_WORDS)
    words[101] = [150]  # SOC over 100 %
    record = await _sweep(tmp_path, monkeypatch, edge_db, words)
    by_id = {entry["id"]: entry for entry in record["registers"]}

    assert by_id["bat_soc"]["value"] == 150.0
    assert by_id["bat_soc"]["quality"] == 1
    assert by_id["bat_soc"]["outside_declared_bounds"] == {"min": 0, "max": 100}
    assert record["totals"]["suspect"] == 1


@pytest.mark.asyncio
async def test_identity_strings_are_captured_and_are_not_registers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    record = await _sweep(tmp_path, monkeypatch, edge_db)
    assert record["identity"] == {"serial": "ABCD"}
    assert "serial" not in {entry["id"] for entry in record["registers"]}


# ── The record is honest about what it cannot show ──────────────


@pytest.mark.asyncio
async def test_the_record_never_promotes_the_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    record = await _sweep(tmp_path, monkeypatch, edge_db)
    assert record["verification"] == "community"
    assert "does not promote" in record["verification_note"]


@pytest.mark.asyncio
async def test_one_sweep_cannot_close_the_hardware_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    record = await _sweep(tmp_path, monkeypatch, edge_db)
    pending = record["pending_checks"]
    assert "counter_monotonicity" in pending
    assert "sign_conventions" in pending
    assert "display_comparison" in pending
    for note in pending.values():
        assert note  # each says what is still missing, not just that it is


@pytest.mark.asyncio
async def test_the_operator_block_is_present_and_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    """Visible blanks: a record with nobody's name on it is not a bench record."""
    record = await _sweep(tmp_path, monkeypatch, edge_db)
    operator = record["operator"]
    assert set(operator) >= {"name", "wiring_port", "wiring_pins", "device_serial"}
    assert all(value is None for value in operator.values())


@pytest.mark.asyncio
async def test_the_record_names_the_map_revision_and_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    record = await _sweep(tmp_path, monkeypatch, edge_db)
    assert record["map"]["name"] == "probe_map"
    assert record["map"]["source"] == "synthetic fixture (unit test) @ rev 0"
    assert record["schema"] == "gen318.vendor_map_validation.v1"


# ── Persistence and the tier-3 summary ──────────────────────────


@pytest.mark.asyncio
async def test_the_record_is_stored_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    await _sweep(tmp_path, monkeypatch, edge_db)
    stored = edge_db.get_vendor_map_validations()

    assert len(stored) == 1
    assert stored[0]["map_name"] == "probe_map"
    assert stored[0]["verification"] == "community"
    assert stored[0]["record"]["totals"]["ok"] == 4


@pytest.mark.asyncio
async def test_each_sweep_appends_rather_than_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    """A record describes one unit at one moment; a later sweep is a new one."""
    await _sweep(tmp_path, monkeypatch, edge_db)
    await _sweep(tmp_path, monkeypatch, edge_db)
    assert len(edge_db.get_vendor_map_validations(map_name="probe_map")) == 2


@pytest.mark.asyncio
async def test_exactly_one_tier_three_summary_per_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    await _sweep(tmp_path, monkeypatch, edge_db)

    items = edge_db.get_queue_items(priority=3, limit=10)
    assert len(items) == 1
    payload = items[0]["payload"]
    assert payload["type"] == "edge.vendor_map_validation"
    assert payload["map"] == "probe_map"
    assert payload["verification"] == "community"
    assert payload["identity"] == {"serial": "ABCD"}
    assert payload["totals"]["ok"] == 4
    # Evidence never competes with alerts or telemetry for a 2G link.
    assert edge_db.get_queue_items(priority=0, limit=10) == []
    assert edge_db.get_queue_items(priority=1, limit=10) == []
    assert edge_db.get_queue_items(priority=2, limit=10) == []


@pytest.mark.asyncio
async def test_the_per_register_detail_stays_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    """A full register dump on a backfill tier is what the ladder prevents."""
    await _sweep(tmp_path, monkeypatch, edge_db)
    payload = edge_db.get_queue_items(priority=3, limit=10)[0]["payload"]
    assert "registers" not in payload
    assert "locally" in payload["detail"]


@pytest.mark.asyncio
async def test_a_device_that_will_not_connect_still_leaves_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    """ "The probe could not reach it" is itself the finding a technician needs."""
    record = await _sweep(tmp_path, monkeypatch, edge_db, connect=False)

    assert record["connected"] is False
    assert record["registers"] == []
    assert len(edge_db.get_vendor_map_validations()) == 1


@pytest.mark.asyncio
async def test_the_stored_record_holds_no_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    _install(tmp_path, monkeypatch)
    probe = VendorMapProbe(
        db=edge_db,
        vendor_map="probe_map",
        transport={
            "protocol": "tcp",
            "host": "192.168.1.10",
            "password": "hunter2",
            "api_token": "secret",
        },
    )
    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        MockClient.return_value = _client(GOOD_WORDS)
        record = await probe.run_once(now_ms=1)

    assert "password" not in record["transport"]
    assert "api_token" not in record["transport"]
    assert record["transport"]["host"] == "192.168.1.10"


# ── Refusals ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_template_refuses_to_be_probed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    """A half-filled ComAp export must not sweep and report a clean run."""
    with pytest.raises(VendorMapError, match="template"):
        await _sweep(tmp_path, monkeypatch, edge_db, text=TEMPLATE_MAP)
    assert edge_db.get_vendor_map_validations() == []


@pytest.mark.asyncio
async def test_an_unknown_map_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    _install(tmp_path, monkeypatch)
    probe = VendorMapProbe(db=edge_db, vendor_map="nope", transport={"host": "h"})
    with pytest.raises(VendorMapError, match="not found"):
        await probe.run_once()


@pytest.mark.asyncio
async def test_the_probe_never_writes_a_register(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    _install(tmp_path, monkeypatch)
    client = _client(GOOD_WORDS)
    probe = VendorMapProbe(
        db=edge_db, vendor_map="probe_map", transport={"protocol": "tcp", "host": "h"}
    )
    with patch("collector.adapters.generic_modbus.AsyncModbusTcpClient") as MockClient:
        MockClient.return_value = client
        await probe.run_once(now_ms=1)

    client.write_register.assert_not_called()
    client.write_registers.assert_not_called()


@pytest.mark.asyncio
async def test_the_probe_writes_no_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge_db: Any
) -> None:
    """Evidence is not telemetry; a sweep must not appear as site data."""
    await _sweep(tmp_path, monkeypatch, edge_db)
    assert edge_db.get_latest_readings() == []


# ── The CLI surface ─────────────────────────────────────────────


def test_the_cli_requires_a_vendor_map() -> None:
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args([])


def test_the_cli_passes_through_only_what_was_given() -> None:
    """Options not given stay absent so the map's own defaults survive."""
    args = _build_arg_parser().parse_args(
        ["--vendor-map", "deye_sg04lp3", "--protocol", "rtu", "--port", "/dev/ttyUSB0"]
    )
    transport = _transport_from_args(args)

    assert transport == {"protocol": "rtu", "port": "/dev/ttyUSB0"}
    assert "baudrate" not in transport
    assert "unit_id" not in transport


def test_the_cli_reads_a_tcp_port_as_a_number() -> None:
    args = _build_arg_parser().parse_args(
        [
            "--vendor-map",
            "m",
            "--protocol",
            "tcp",
            "--host",
            "10.0.0.2",
            "--port",
            "1502",
        ]
    )
    transport = _transport_from_args(args)
    assert transport["port"] == 1502
    assert transport["host"] == "10.0.0.2"


def test_the_cli_accepts_the_serial_framing_options() -> None:
    args = _build_arg_parser().parse_args(
        [
            "--vendor-map",
            "dse_gencomm",
            "--protocol",
            "rtu",
            "--port",
            "/dev/ttyUSB0",
            "--baudrate",
            "115200",
            "--parity",
            "E",
            "--stopbits",
            "2",
            "--bytesize",
            "7",
            "--unit-id",
            "10",
            "--request-spacing-ms",
            "20",
        ]
    )
    transport = _transport_from_args(args)
    assert transport["baudrate"] == 115200
    assert transport["parity"] == "E"
    assert transport["stopbits"] == 2
    assert transport["bytesize"] == 7
    assert transport["unit_id"] == 10
    assert transport["request_spacing_ms"] == 20


def test_the_cli_exposes_no_write_option() -> None:
    parser = _build_arg_parser()
    options = {action.dest for action in parser._actions}
    assert not any("write" in option or "set" == option for option in options)
    assert "vendor_map" in options
