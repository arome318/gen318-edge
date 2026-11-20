"""Tests for OkraHarvestAdapter (Doc 15 §7.5, F-HW-08)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from collector.adapters.base import AdapterStatus
from collector.adapters.okra_harvest import OkraHarvestAdapter

HUB_UUID = "6f1c2a7e-3b4d-4e5f-8a9b-0c1d2e3f4a5b"
NODE_A_UUID = "0b8e7d6c-5a4f-4e3d-9c2b-1a0f9e8d7c6b"
NODE_B_UUID = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

OKRA_CONFIG: dict = {
    "api_url": "https://api.okrasolar.com/v1",
    "api_key": "test-api-key",
    "site_id": "okra-site-001",
    "poll_interval_seconds": 300,
    "asset_mappings": {
        "hub": HUB_UUID,
        "nodes": {
            "node-a": NODE_A_UUID,
            "node-b": NODE_B_UUID,
        },
    },
}

TOPOLOGY_RESPONSE = {
    "nodes": [
        {"id": "node-a", "lat": 9.06, "lng": 7.49},
        {"id": "node-b", "lat": 9.07, "lng": 7.50},
        {"id": "node-c", "lat": 9.08, "lng": 7.51},
    ],
    "connections": [
        {"from": "node-a", "to": "node-b"},
        {"from": "node-b", "to": "node-c"},
    ],
}

NODES_RESPONSE = {
    "nodes": [
        {
            "id": "node-a",
            "solar_power_w": 450.5,
            "battery_soc_pct": 78.0,
            "load_power_w": 200.0,
            "power_sharing_w": 50.0,
        },
        {
            "id": "node-b",
            "solar_power_w": 320.0,
            "battery_soc_pct": 65.0,
            "load_power_w": 180.0,
        },
        {
            "id": "node-c",
            "solar_power_w": 0.0,
            "battery_soc_pct": 20.0,
        },
    ],
}


def _mock_response(
    status_code: int = 200, json_data: dict | None = None
) -> httpx.Response:
    """Build a mock httpx.Response."""
    response = httpx.Response(
        status_code=status_code,
        json=json_data or {},
        request=httpx.Request("GET", "https://test.com"),
    )
    return response


# ── Initialize Tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_initialize_success() -> None:
    """Successful API validation sets CONNECTED status."""
    adapter = OkraHarvestAdapter()

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _mock_response(200, {"sites": []})
        result = await adapter.initialize(OKRA_CONFIG)

    assert result is True
    assert adapter._status == AdapterStatus.CONNECTED


@pytest.mark.asyncio
async def test_initialize_api_failure() -> None:
    """API validation failure sets ERROR status."""
    adapter = OkraHarvestAdapter()

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        result = await adapter.initialize(OKRA_CONFIG)

    assert result is False
    assert adapter._status == AdapterStatus.ERROR


# ── Poll Tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_poll_topology_and_nodes() -> None:
    """Poll extracts topology metrics and per-node metrics."""
    adapter = OkraHarvestAdapter()

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        # Initialize
        mock_get.return_value = _mock_response(200, {"sites": []})
        await adapter.initialize(OKRA_CONFIG)

        # Poll - returns topology first, then nodes
        mock_get.side_effect = [
            _mock_response(200, TOPOLOGY_RESPONSE),
            _mock_response(200, NODES_RESPONSE),
        ]
        readings = await adapter.poll()

    # node-c has no mapping and is skipped entirely:
    # 2 topology + 2 solar + 2 soc + 2 load + 1 sharing = 9
    assert len(readings) == 9

    # Check topology metrics
    topo_readings = [r for r in readings if r.metric_name.startswith("mesh.")]
    node_counts = [r for r in topo_readings if r.metric_name == "mesh.node_count"]
    assert len(node_counts) == 1
    assert node_counts[0].value == 3.0
    assert node_counts[0].asset_id == HUB_UUID

    conn_counts = [
        r for r in topo_readings if r.metric_name == "mesh.active_connections"
    ]
    assert len(conn_counts) == 1
    assert conn_counts[0].value == 2.0

    # Check per-node solar
    solar_readings = [r for r in readings if r.metric_name == "solar.pv_power_w"]
    assert len(solar_readings) == 2
    # node-a should map to configured asset
    node_a_solar = [r for r in solar_readings if r.asset_id == NODE_A_UUID]
    assert len(node_a_solar) == 1
    assert node_a_solar[0].value == 450.5

    # No reading carries an invented id: every asset_id is a real UUID, so
    # nothing is collected only to be rejected by the sync pre-flight.
    from sync.engine import _is_uuid

    assert all(_is_uuid(r.asset_id) for r in readings)


@pytest.mark.asyncio
async def test_poll_power_sharing() -> None:
    """Power sharing metric only appears when node reports it."""
    adapter = OkraHarvestAdapter()

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _mock_response(200, {"sites": []})
        await adapter.initialize(OKRA_CONFIG)

        mock_get.side_effect = [
            _mock_response(200, TOPOLOGY_RESPONSE),
            _mock_response(200, NODES_RESPONSE),
        ]
        readings = await adapter.poll()

    sharing = [r for r in readings if r.metric_name == "mesh.power_sharing_w"]
    assert len(sharing) == 1
    assert sharing[0].value == 50.0
    assert sharing[0].asset_id == NODE_A_UUID


@pytest.mark.asyncio
async def test_poll_unmapped_hub_skips_topology_but_keeps_mapped_nodes() -> None:
    """No hub mapping: topology metrics are skipped, not given a fake id."""
    adapter = OkraHarvestAdapter()
    config = {
        **OKRA_CONFIG,
        "asset_mappings": {"nodes": {"node-a": NODE_A_UUID}},
    }

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _mock_response(200, {"sites": []})
        await adapter.initialize(config)
        mock_get.side_effect = [
            _mock_response(200, TOPOLOGY_RESPONSE),
            _mock_response(200, NODES_RESPONSE),
        ]
        readings = await adapter.poll()

    assert not [r for r in readings if r.metric_name.startswith("mesh.node")]
    assert {r.asset_id for r in readings} == {NODE_A_UUID}
    # The topology itself is still kept for local use.
    assert len(adapter._topology["nodes"]) == 3


@pytest.mark.asyncio
async def test_poll_non_uuid_mapping_is_skipped() -> None:
    """A mapping that is not a UUID is treated as unmapped, not passed through."""
    adapter = OkraHarvestAdapter()
    config = {
        **OKRA_CONFIG,
        "asset_mappings": {"hub": "okra-hub-1", "nodes": {"node-a": "okra-node-a"}},
    }

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _mock_response(200, {"sites": []})
        await adapter.initialize(config)
        mock_get.side_effect = [
            _mock_response(200, TOPOLOGY_RESPONSE),
            _mock_response(200, NODES_RESPONSE),
        ]
        readings = await adapter.poll()

    assert readings == []


@pytest.mark.asyncio
async def test_unmapped_node_warns_once_per_id() -> None:
    """Repeated polls do not re-log the same unmapped vendor id."""
    adapter = OkraHarvestAdapter()

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _mock_response(200, {"sites": []})
        await adapter.initialize(OKRA_CONFIG)
        with patch("collector.adapters.okra_harvest.logger") as mock_logger:
            for _ in range(3):
                mock_get.side_effect = [
                    _mock_response(200, TOPOLOGY_RESPONSE),
                    _mock_response(200, NODES_RESPONSE),
                ]
                await adapter.poll()

    unmapped = [
        call
        for call in mock_logger.warning.call_args_list
        if call.args and call.args[0] == "okra_asset_unmapped"
    ]
    assert len(unmapped) == 1
    assert unmapped[0].kwargs["vendor_id"] == "node:node-c"


@pytest.mark.asyncio
async def test_poll_connect_error() -> None:
    """ConnectError sets ERROR status, returns empty."""
    adapter = OkraHarvestAdapter()

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _mock_response(200, {"sites": []})
        await adapter.initialize(OKRA_CONFIG)

        mock_get.side_effect = httpx.ConnectError("Network unreachable")
        readings = await adapter.poll()

    assert readings == []
    assert adapter._status == AdapterStatus.ERROR


@pytest.mark.asyncio
async def test_poll_timeout_error() -> None:
    """TimeoutException sets ERROR status, returns empty."""
    adapter = OkraHarvestAdapter()

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _mock_response(200, {"sites": []})
        await adapter.initialize(OKRA_CONFIG)

        mock_get.side_effect = httpx.TimeoutException("Read timeout")
        readings = await adapter.poll()

    assert readings == []
    assert adapter._status == AdapterStatus.ERROR


@pytest.mark.asyncio
async def test_poll_no_client() -> None:
    """Returns empty if client is not initialized."""
    adapter = OkraHarvestAdapter()
    readings = await adapter.poll()
    assert readings == []


# ── Health Check Tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_check_connected() -> None:
    adapter = OkraHarvestAdapter()

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _mock_response(200, {"sites": []})
        await adapter.initialize(OKRA_CONFIG)

    status = await adapter.health_check()
    assert status == AdapterStatus.CONNECTED


@pytest.mark.asyncio
async def test_health_check_error_stale() -> None:
    """Health check returns ERROR after 10 minutes without readings."""
    adapter = OkraHarvestAdapter()

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _mock_response(200, {"sites": []})
        await adapter.initialize(OKRA_CONFIG)

    adapter._last_reading_at = time.time() - 700  # >10 min

    status = await adapter.health_check()
    assert status == AdapterStatus.ERROR


@pytest.mark.asyncio
async def test_health_check_disconnected() -> None:
    adapter = OkraHarvestAdapter()
    status = await adapter.health_check()
    assert status == AdapterStatus.DISCONNECTED


# ── Capabilities Tests ─────────────────────────────────────────


def test_capabilities() -> None:
    adapter = OkraHarvestAdapter()
    adapter._config = OKRA_CONFIG

    caps = adapter.capabilities()

    assert caps.can_read is True
    assert caps.can_write is False
    assert caps.can_subscribe is False
    assert "mesh.node_count" in caps.supported_metrics
    assert "mesh.power_sharing_w" in caps.supported_metrics
    assert caps.poll_interval_seconds == 300
    assert caps.min_poll_interval_seconds == 60


# ── Write / Shutdown Tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_write_returns_false() -> None:
    adapter = OkraHarvestAdapter()
    assert await adapter.write("solar.pv_power_w", 100.0) is False


@pytest.mark.asyncio
async def test_shutdown() -> None:
    adapter = OkraHarvestAdapter()

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = _mock_response(200, {"sites": []})
        await adapter.initialize(OKRA_CONFIG)

    with patch.object(
        httpx.AsyncClient, "aclose", new_callable=AsyncMock
    ) as mock_close:
        await adapter.shutdown()
        mock_close.assert_called_once()

    assert adapter._status == AdapterStatus.DISCONNECTED
