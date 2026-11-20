"""The opt-in CAN bridge process.

A SocketCAN interface lives in a network namespace and no Docker feature
hands a container one host interface while it keeps its own, so reading a
CAN tap from a container means sharing the host's namespace. This process
exists so that only IT has to, rather than the collector — which loads every
vendor adapter and its credentials.

It writes straight to the same SQLite file the collector writes, because
`inference` is the only subscriber to the telemetry topic and nothing writes
that topic to storage. A bridge that only published would reach local
inference and never leave the node.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from collector.adapters.base import Reading
from collector.can_bridge import CANBridge, adapter_config

BATTERY = "a0000000-0000-0000-0000-000000000001"


def _config(**extra: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "interface": "can0",
        "bitrate": 500000,
        "profile": "victron_generic",
        "asset_id": BATTERY,
        "require_listen_only": True,
    }
    config.update(extra)
    return config


def _readings() -> list[Reading]:
    now = datetime.now(UTC)
    return [
        Reading("battery.soc_pct", 88.0, now, 0, "can:0x355:0", BATTERY),
        Reading("battery.voltage_v", 51.2, now, 0, "can:0x356:0", BATTERY),
    ]


# ── Configuration from the environment ──────────────────────────


def test_config_defaults_to_the_victron_profile_at_500k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "EDGE_CAN_INTERFACE",
        "EDGE_CAN_BITRATE",
        "EDGE_CAN_PROFILE",
        "EDGE_CAN_ASSET_ID",
        "EDGE_CAN_REQUIRE_LISTEN_ONLY",
    ):
        monkeypatch.delenv(name, raising=False)

    config = adapter_config()
    assert config["interface"] == "can0"
    assert config["bitrate"] == 500000
    assert config["profile"] == "victron_generic"
    assert config["asset_id"] == ""  # no default: it must be named


def test_the_listen_only_gate_defaults_to_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two active masters on a BMS bus is how a battery disconnects itself."""
    monkeypatch.delenv("EDGE_CAN_REQUIRE_LISTEN_ONLY", raising=False)
    assert adapter_config()["require_listen_only"] is True


def test_the_gate_can_be_opened_only_by_setting_it_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EDGE_CAN_REQUIRE_LISTEN_ONLY", "false")
    assert adapter_config()["require_listen_only"] is False
    monkeypatch.setenv("EDGE_CAN_REQUIRE_LISTEN_ONLY", "true")
    assert adapter_config()["require_listen_only"] is True


def test_the_ingeteam_bitrate_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGE_CAN_BITRATE", "250000")
    assert adapter_config()["bitrate"] == 250000


# ── Readings reach SQLite ───────────────────────────────────────


@pytest.mark.asyncio
async def test_readings_are_written_to_the_shared_database(edge_db: Any) -> None:
    """The durable path. Publishing alone would never leave the node."""
    bridge = CANBridge(db=edge_db, config=_config())
    bridge._store(_readings())

    stored = {row["metric_name"]: row["value"] for row in edge_db.get_latest_readings()}
    assert stored["battery.soc_pct"] == 88.0
    assert stored["battery.voltage_v"] == pytest.approx(51.2)


@pytest.mark.asyncio
async def test_stored_rows_start_unsynced_so_sync_will_drain_them(edge_db: Any) -> None:
    bridge = CANBridge(db=edge_db, config=_config())
    bridge._store(_readings())
    assert edge_db.count_unsynced_measurements() == 2


@pytest.mark.asyncio
async def test_rows_carry_the_bound_battery_asset(edge_db: Any) -> None:
    bridge = CANBridge(db=edge_db, config=_config())
    bridge._store(_readings())
    rows = edge_db.get_unsynced_measurements(limit=10)
    assert {row["asset_id"] for row in rows} == {BATTERY}


@pytest.mark.asyncio
async def test_writing_the_same_reading_twice_is_idempotent(edge_db: Any) -> None:
    """Measurements are keyed (timestamp, asset, metric) with INSERT OR IGNORE."""
    bridge = CANBridge(db=edge_db, config=_config())
    readings = _readings()
    bridge._store(readings)
    bridge._store(readings)
    assert edge_db.count_unsynced_measurements() == 2


# ── MQTT is best effort, and second ─────────────────────────────


@pytest.mark.asyncio
async def test_readings_are_also_published_for_local_inference(edge_db: Any) -> None:
    mqtt = MagicMock()
    bridge = CANBridge(db=edge_db, config=_config(), mqtt=mqtt, device_id="edge-1")
    bridge._store(_readings())

    assert mqtt.publish.call_count == 2
    topic, payload = mqtt.publish.call_args.args
    assert topic == "gen318/telemetry"
    assert payload["asset_id"] == BATTERY
    assert payload["device_id"] == "edge-1"


@pytest.mark.asyncio
async def test_a_broker_outage_never_costs_a_reading(edge_db: Any) -> None:
    """Storage happens first, so a publish failure cannot lose data."""
    mqtt = MagicMock()
    mqtt.publish.side_effect = RuntimeError("broker gone")
    bridge = CANBridge(db=edge_db, config=_config(), mqtt=mqtt)

    bridge._store(_readings())  # must not raise

    assert edge_db.count_unsynced_measurements() == 2


@pytest.mark.asyncio
async def test_the_bridge_runs_without_a_broker_at_all(edge_db: Any) -> None:
    bridge = CANBridge(db=edge_db, config=_config(), mqtt=None)
    bridge._store(_readings())
    assert edge_db.count_unsynced_measurements() == 2


@pytest.mark.asyncio
async def test_a_storage_failure_is_logged_and_does_not_raise(edge_db: Any) -> None:
    """Offline-first: nothing in this process may crash the container."""
    broken = MagicMock()
    broken.insert_measurements.side_effect = RuntimeError("disk gone")
    mqtt = MagicMock()
    bridge = CANBridge(db=broken, config=_config(), mqtt=mqtt)

    bridge._store(_readings())  # must not raise

    # And it does not publish what it could not store.
    mqtt.publish.assert_not_called()


# ── The run loop ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_refused_bus_waits_rather_than_hammering(edge_db: Any) -> None:
    """A bus that is not listen-only needs a person, not a retry storm."""
    slept: list[float] = []
    bridge = CANBridge(db=edge_db, config=_config())

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)
        bridge.stop()  # one pass only

    with (
        patch.object(CANBridge, "_sleep", AsyncMock(side_effect=_sleep)),
        patch(
            "collector.can_bridge.CANBMSListenerAdapter.initialize",
            AsyncMock(return_value=False),
        ),
    ):
        await bridge.run()

    assert slept == [60.0]


@pytest.mark.asyncio
async def test_an_exception_during_initialize_does_not_escape(edge_db: Any) -> None:
    bridge = CANBridge(db=edge_db, config=_config())

    async def _sleep(seconds: float) -> None:
        bridge.stop()

    with (
        patch.object(CANBridge, "_sleep", AsyncMock(side_effect=_sleep)),
        patch(
            "collector.can_bridge.CANBMSListenerAdapter.initialize",
            AsyncMock(side_effect=RuntimeError("bus exploded")),
        ),
    ):
        await bridge.run()  # must not raise


@pytest.mark.asyncio
async def test_a_poll_failure_does_not_end_the_loop(edge_db: Any) -> None:
    bridge = CANBridge(db=edge_db, config=_config(), poll_interval_seconds=0.01)
    polls = 0

    async def _poll() -> list[Reading]:
        nonlocal polls
        polls += 1
        if polls == 1:
            raise RuntimeError("transceiver hiccup")
        bridge.stop()
        return _readings()

    with (
        patch(
            "collector.can_bridge.CANBMSListenerAdapter.initialize",
            AsyncMock(return_value=True),
        ),
        patch(
            "collector.can_bridge.CANBMSListenerAdapter.poll",
            AsyncMock(side_effect=_poll),
        ),
        patch("collector.can_bridge.CANBMSListenerAdapter.shutdown", AsyncMock()),
    ):
        await bridge.run()

    assert polls == 2
    assert edge_db.count_unsynced_measurements() == 2


@pytest.mark.asyncio
async def test_the_adapter_is_shut_down_when_the_loop_ends(edge_db: Any) -> None:
    bridge = CANBridge(db=edge_db, config=_config(), poll_interval_seconds=0.01)
    shutdown = AsyncMock()

    async def _poll() -> list[Reading]:
        bridge.stop()
        return []

    with (
        patch(
            "collector.can_bridge.CANBMSListenerAdapter.initialize",
            AsyncMock(return_value=True),
        ),
        patch(
            "collector.can_bridge.CANBMSListenerAdapter.poll",
            AsyncMock(side_effect=_poll),
        ),
        patch("collector.can_bridge.CANBMSListenerAdapter.shutdown", shutdown),
    ):
        await bridge.run()

    shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancellation_propagates_cleanly(edge_db: Any) -> None:
    bridge = CANBridge(db=edge_db, config=_config(), poll_interval_seconds=10)

    with (
        patch(
            "collector.can_bridge.CANBMSListenerAdapter.initialize",
            AsyncMock(return_value=True),
        ),
        patch(
            "collector.can_bridge.CANBMSListenerAdapter.poll",
            AsyncMock(return_value=[]),
        ),
        patch("collector.can_bridge.CANBMSListenerAdapter.shutdown", AsyncMock()),
    ):
        task = asyncio.create_task(bridge.run())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ── The compose service it runs as ──────────────────────────────


def _compose() -> dict[str, Any]:
    return yaml.safe_load(Path("../docker-compose.yml").read_text())


def test_the_bridge_is_opt_in() -> None:
    """A node with no CAN tap must not run it, or log a refusal forever."""
    service = _compose()["services"]["can-bridge"]
    assert service["profiles"] == ["can"]


def test_the_bridge_shares_the_hosts_network_namespace() -> None:
    """The only in-container way to see a SocketCAN interface."""
    assert _compose()["services"]["can-bridge"]["network_mode"] == "host"


def test_no_other_service_takes_host_networking() -> None:
    """The point of the sidecar: confine that to one small process."""
    for name, service in _compose()["services"].items():
        if name == "can-bridge":
            continue
        assert service.get("network_mode") != "host", name


def test_the_bridge_writes_the_same_database_the_collector_does() -> None:
    service = _compose()["services"]["can-bridge"]
    assert "collector_data:/data" in service["volumes"]
    # Read-write, like the collector and sync — not the read-only mount
    # inference and api take.
    assert "collector_data:/data:ro" not in service["volumes"]


def test_the_bridge_reaches_the_broker_on_loopback() -> None:
    """Under host networking the compose service name no longer resolves."""
    environment = _compose()["services"]["can-bridge"]["environment"]
    assert "MQTT_HOST=127.0.0.1" in environment


def test_the_bridge_runs_the_collector_image_with_its_own_command() -> None:
    """No second image to build or keep in step."""
    service = _compose()["services"]["can-bridge"]
    assert service["build"]["dockerfile"] == "collector/Dockerfile.arm64"
    assert service["command"] == ["python", "-m", "collector.can_bridge"]


def test_the_asset_id_has_no_default_in_compose() -> None:
    """An unattributed battery reading has nowhere to land."""
    environment = _compose()["services"]["can-bridge"]["environment"]
    assert "EDGE_CAN_ASSET_ID=${EDGE_CAN_ASSET_ID:-}" in environment


def test_the_listen_only_gate_defaults_on_in_compose_too() -> None:
    environment = _compose()["services"]["can-bridge"]["environment"]
    assert (
        "EDGE_CAN_REQUIRE_LISTEN_ONLY=${EDGE_CAN_REQUIRE_LISTEN_ONLY:-true}"
        in environment
    )
