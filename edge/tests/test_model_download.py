from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from common.model_manifest import canonical_manifest_bytes
from sync.engine import SyncConfig, SyncEngine


@pytest.fixture
def sync_engine(edge_db) -> SyncEngine:
    return SyncEngine(
        SyncConfig(
            cloud_api_url="https://api.gen318.io",
            device_id="test-edge-001",
            device_cert_path="/tmp/missing.pem",
            device_key_path="/tmp/missing.key",
            ca_cert_path="/tmp/missing-ca.pem",
            cert_pin_sha256="",
        ),
        edge_db,
    )


def _manifest(payload: bytes) -> tuple[dict, dict[str, str]]:
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    manifest = {
        "update_id": "0155b28e-ff99-4b1e-856a-9cc26ea590e4",
        "model_name": "battery_health",
        "version": "3",
        "artifact_uri": "s3://models/model.ubj",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "target_nodes": ["*"],
        "rollout_pct": 100,
        "strategy": "stable",
        "timestamp": 1,
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        "key_id": "test-key",
        "download_url": "https://objects.example/presigned",
    }
    contract = {
        "schema_version": 1,
        "model_name": "battery_health",
        "feature_names": ["f1"],
        "feature_count": 1,
        "missing_value_policy": "zero",
        "output_kind": "regression",
    }
    manifest["feature_contract"] = contract
    manifest["golden_vectors"] = [{"inputs": {"f1": 1.0}, "expected": [2.0]}]
    manifest["contract_sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["signature"] = base64.b64encode(
        private.sign(canonical_manifest_bytes(manifest))
    ).decode()
    return manifest, {"test-key": base64.b64encode(public).decode()}


class _Response:
    status_code = 200

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    async def aiter_bytes(self):
        yield self.payload[:3]
        yield self.payload[3:]


class _Context:
    def __init__(self, value) -> None:
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *args) -> None:
        return None


class _Client:
    def __init__(self, payload: bytes, **_kwargs) -> None:
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        return None

    def stream(self, method: str, url: str) -> _Context:
        assert method == "GET"
        assert url.startswith("https://")
        return _Context(_Response(self.payload))


@pytest.mark.asyncio
async def test_download_verifies_streams_and_atomically_stages(
    sync_engine, tmp_path, monkeypatch
) -> None:
    payload = b"signed-xgboost-artifact"
    manifest, keys = _manifest(payload)
    sync_engine.config.model_path = str(tmp_path)
    sync_engine._trusted_model_keys = keys
    monkeypatch.setattr(
        "sync.engine.httpx.AsyncClient", lambda **kwargs: _Client(payload, **kwargs)
    )

    staged = await sync_engine._download_model(manifest)

    assert staged is not None
    assert staged["path"].endswith("-battery_health.ubj")
    assert (
        tmp_path / "staging" / staged["path"].split("/")[-1]
    ).read_bytes() == payload
    assert not list((tmp_path / "staging").glob("*.partial"))


@pytest.mark.asyncio
async def test_download_rejects_tampered_manifest_before_network(
    sync_engine, tmp_path, monkeypatch
) -> None:
    manifest, keys = _manifest(b"model")
    manifest["size_bytes"] = 6
    sync_engine.config.model_path = str(tmp_path)
    sync_engine._trusted_model_keys = keys

    def _network_must_not_run(**_kwargs):
        raise AssertionError("network used for an invalid signature")

    monkeypatch.setattr("sync.engine.httpx.AsyncClient", _network_must_not_run)
    assert await sync_engine._download_model(manifest) is None
    queued = sync_engine.db.get_queue_items(priority=1)
    assert queued[0]["payload"]["status"] == "download_failed"
    assert queued[0]["payload"]["model_results"] == {
        "battery_health": "invalid_signature"
    }


@pytest.mark.asyncio
async def test_partial_bundle_keeps_failed_manifest_in_activation_contract(
    sync_engine,
) -> None:
    first = {"model_name": "battery_health"}
    second = {"model_name": "demand_forecast"}
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "update_available": True,
        "firmware_update": True,
        "target_version": "2.0.0",
        "channel": "stable",
        "models": [first, second],
    }
    sync_engine._http_client.get = AsyncMock(return_value=response)
    sync_engine._download_model = AsyncMock(
        side_effect=[{**first, "path": "/models/first.ubj"}, None]
    )
    mqtt = MagicMock()

    await sync_engine.check_for_ota(mqtt)

    published = mqtt.publish.call_args.args[1]
    assert len(published["models"]) == 2
    assert published["models"][0]["path"] == "/models/first.ubj"
    assert published["models"][1] == second
