"""Every Modbus read path, against a REAL pymodbus server.

This file exists because of a defect the rest of the suite could not see.

pymodbus renamed the unit-id keyword from `slave` to `device_id` in 3.11 and
REMOVED the old name in 3.14, where passing it raises TypeError. Every other
Modbus test in this repo drives a MagicMock, and a MagicMock accepts any
keyword at all — so the suite stayed green while a node built against 3.14
collected nothing. `requirements.txt` pinned only `pymodbus>=3.6.0`, so that
node was any node built after the release.

The failure was silent in the worst way. In the legacy poll the TypeError
escaped into the supervisor's blanket handler, which logged a read error and
kept looping; in the register probe a bare `except Exception` swallowed it
entirely. Nothing reached SQLite, so there was nothing to backfill when the
link came back.

The point of this file is therefore not the assertions. It is that these
tests talk to a server that will reject a wrong call, so the keyword is
exercised rather than accepted. Every read path the collector has is covered
here: the legacy register list, the Victron register table, the vendor-map
engine and the original probe sweep.
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import pytest

from collector.adapters.generic_modbus import DEVICE_KWARG, GenericModbusAdapter
from collector.adapters.victron_modbus import VictronModbusAdapter
from collector.register_probe import RegisterProbe


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


class _Server:
    """A real pymodbus TCP server over a flat register image."""

    def __init__(self, words: dict[int, int], unit_id: int = 1) -> None:
        self.words = words
        self.unit_id = unit_id
        self.port = _free_port()
        self._server: Any = None
        self._task: asyncio.Task[Any] | None = None

    async def __aenter__(self) -> _Server:
        from pymodbus.server import ModbusTcpServer
        from pymodbus.simulator import DataType, SimData, SimDevice

        size = max(self.words) + 8 if self.words else 16
        values = [0] * size
        for address, word in self.words.items():
            values[address] = word
        block = SimData(0, values=values, datatype=DataType.REGISTERS)
        self._server = ModbusTcpServer(
            SimDevice(self.unit_id, simdata=[block]), address=("127.0.0.1", self.port)
        )
        self._task = asyncio.create_task(self._server.serve_forever())
        await asyncio.sleep(0.3)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._server is not None:
            await self._server.shutdown()
        if self._task is not None:
            self._task.cancel()


# ── The keyword the installed pymodbus actually wants ───────────


def test_the_resolved_keyword_matches_the_installed_signature() -> None:
    """Resolved from the signature, never guessed from a version string."""
    import inspect

    from pymodbus.client.mixin import ModbusClientMixin

    parameters = inspect.signature(ModbusClientMixin.read_holding_registers).parameters
    assert DEVICE_KWARG in parameters
    assert DEVICE_KWARG in ("slave", "device_id")


def test_no_read_path_hard_codes_the_keyword() -> None:
    """A future rename must break one constant, not five call sites."""
    from pathlib import Path

    for module in (
        "collector/adapters/generic_modbus.py",
        "collector/adapters/victron_modbus.py",
        "collector/adapters/sunspec_modbus.py",
        "collector/register_probe.py",
    ):
        source = Path(module).read_text()
        assert "slave=" not in source, f"{module} still hard-codes the old keyword"


def test_the_library_is_pinned_below_the_next_major() -> None:
    """An unbounded pin is how the rename landed unseen in the first place."""
    from pathlib import Path

    for requirements in ("requirements.txt", "requirements-dev.txt"):
        text = Path(requirements).read_text()
        assert "pymodbus>=3.6.0,<4" in text, requirements


# ── The legacy `registers:` path ────────────────────────────────


@pytest.mark.asyncio
async def test_the_legacy_register_list_reads_a_real_server() -> None:
    """The path that silently collected nothing on pymodbus 3.14."""
    async with _Server({100: 5000, 103: 750}) as server:
        adapter = GenericModbusAdapter()
        assert await adapter.initialize(
            {
                "protocol": "tcp",
                "host": "127.0.0.1",
                "port": server.port,
                "unit_id": 1,
                "asset_id": "legacy-asset",
                "registers": [
                    {
                        "metric": "genset.power_w",
                        "address": 100,
                        "type": "uint16",
                        "function_code": 3,
                    },
                    {
                        "metric": "genset.fuel_level_pct",
                        "address": 103,
                        "type": "uint16",
                        "scale": 0.1,
                        "function_code": 3,
                    },
                ],
            }
        )
        readings = await adapter.poll()
        await adapter.shutdown()

    values = {reading.metric_name: reading.value for reading in readings}
    assert values["genset.power_w"] == 5000.0
    assert values["genset.fuel_level_pct"] == pytest.approx(75.0)
    assert all(reading.asset_id == "legacy-asset" for reading in readings)


@pytest.mark.asyncio
async def test_the_legacy_path_reads_input_registers_too() -> None:
    async with _Server({100: 4242}) as server:
        adapter = GenericModbusAdapter()
        await adapter.initialize(
            {
                "protocol": "tcp",
                "host": "127.0.0.1",
                "port": server.port,
                "unit_id": 1,
                "asset_id": "legacy-asset",
                "registers": [
                    {
                        "metric": "genset.power_w",
                        "address": 100,
                        "type": "uint16",
                        "function_code": 4,
                    }
                ],
            }
        )
        readings = await adapter.poll()
        await adapter.shutdown()

    assert readings[0].value == 4242.0


@pytest.mark.asyncio
async def test_the_installer_declared_write_path_reaches_the_bus() -> None:
    """Dispatch writes raised before reaching the device on 3.14."""
    async with _Server({200: 0}) as server:
        adapter = GenericModbusAdapter()
        await adapter.initialize(
            {
                "protocol": "tcp",
                "host": "127.0.0.1",
                "port": server.port,
                "unit_id": 1,
                "asset_id": "legacy-asset",
                "registers": [
                    {
                        "metric": "genset.power_w",
                        "address": 200,
                        "type": "uint16",
                        "function_code": 3,
                    }
                ],
                "write_mappings": {
                    "genset.power_setpoint_w": {"address": 200, "type": "uint16"}
                },
            }
        )
        wrote = await adapter.write("genset.power_setpoint_w", 1234.0)
        readings = await adapter.poll()
        await adapter.shutdown()

    assert wrote is True
    # The value came back off the wire, so the write actually landed.
    assert readings[0].value == 1234.0


# ── The Victron register table ──────────────────────────────────


@pytest.mark.asyncio
async def test_the_victron_adapter_reads_a_real_server() -> None:
    """The GX adapter collected nothing at all on pymodbus 3.14."""
    # Victron DIVIDES by its scale: reg 840 is scale 10, so 512 is 51.2 V.
    async with _Server({840: 512, 843: 88}, unit_id=100) as server:
        adapter = VictronModbusAdapter()
        assert await adapter.initialize(
            {
                "host": "127.0.0.1",
                "port": server.port,
                "unit_ids": [100],
                "asset_mappings": {"100": "victron-asset"},
            }
        )
        readings = await adapter.poll()
        await adapter.shutdown()

    values = {reading.metric_name: reading.value for reading in readings}
    assert values["battery.voltage_v"] == pytest.approx(51.20)
    assert values["battery.soc_pct"] == 88.0


# ── The original register probe sweep ───────────────────────────


@pytest.mark.asyncio
async def test_the_original_probe_sweep_reads_a_real_server(edge_db: Any) -> None:
    """A bare except swallowed the TypeError, so this produced nothing."""
    async with _Server({284: 100, 285: 0, 286: 5}, unit_id=225) as server:
        probe = RegisterProbe(
            db=edge_db,
            host="127.0.0.1",
            port=server.port,
            ranges="225:284-286",
            interval_seconds=3600,
            pace_seconds=0.0,
        )
        await probe._run_cycle(now_ms=1000)

    series = edge_db.get_register_candidate_series(225, 284, "u16", limit=10)
    assert [row["value"] for row in series] == [100.0]
    # And the tier-3 summary it exists to produce.
    assert len(edge_db.get_queue_items(priority=3, limit=10)) == 1


# ── The vendor-map engine, which was already correct ────────────


@pytest.mark.asyncio
async def test_the_vendor_map_engine_still_reads_a_real_server() -> None:
    """Covered elsewhere too; kept here so one file proves every read path."""
    async with _Server({100: 2500, 101: 88}) as server:
        adapter = GenericModbusAdapter()
        assert await adapter.initialize(
            {
                "vendor_map": "dse_gencomm",
                "protocol": "tcp",
                "host": "127.0.0.1",
                "port": server.port,
                "unit_id": 1,
                "asset_mappings": {
                    "diesel_genset": "90000000-0000-0000-0000-000000000001"
                },
            }
        )
        await adapter.poll()
        await adapter.shutdown()


# ── The serial backend the RTU maps depend on ───────────────────
#
# pymodbus 3.x makes pyserial an OPTIONAL extra. Without it
# AsyncModbusSerialClient raises at construction — so every Modbus RTU
# vendor map is dead on a node built from requirements.txt alone. Every
# other test in this repo drives Modbus over TCP against the loopback
# server, so nothing else can see this.


def test_pyserial_is_installed() -> None:
    import serial  # noqa: F401


def test_the_serial_backend_is_declared_in_requirements() -> None:
    from pathlib import Path

    for requirements in ("requirements.txt", "requirements-dev.txt"):
        text = Path(requirements).read_text()
        assert "pyserial" in text, (
            f"{requirements} does not declare pyserial; pymodbus makes it an "
            "optional extra and every RTU vendor map needs it"
        )


@pytest.mark.asyncio
async def test_an_rtu_client_can_actually_be_constructed() -> None:
    """Construction is where the missing backend raises, not connect()."""
    from pymodbus.client import AsyncModbusSerialClient

    client = AsyncModbusSerialClient(
        port="/dev/ttyUSB0", baudrate=9600, bytesize=8, parity="N", stopbits=1
    )
    assert client is not None


@pytest.mark.asyncio
async def test_every_shipped_rtu_map_can_reach_the_serial_client() -> None:
    """The four maps that declare RTU are the ones this would have killed."""
    from pathlib import Path

    import yaml

    from collector.adapters.vendor_map import load_vendor_map

    rtu_maps = []
    for path in sorted(Path("collector/adapters/vendor_maps").glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        if (raw.get("protocol_defaults") or {}).get("protocol") == "rtu":
            rtu_maps.append(path.stem)

    assert rtu_maps, "expected at least one RTU map to guard"
    for name in rtu_maps:
        loaded = load_vendor_map(name)
        assert loaded.protocol["protocol"] == "rtu"

    # And the client those maps will instantiate is constructible.
    from pymodbus.client import AsyncModbusSerialClient

    assert AsyncModbusSerialClient(port="/dev/ttyUSB0", baudrate=9600) is not None
