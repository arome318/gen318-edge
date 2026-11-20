"""OTA contract tests for the inference service.

firmware_version is what check_for_ota() compares against the cloud target
to decide whether an update is still pending — it must be stamped only when
every model in the manifest was verified and accepted, or a failed OTA
self-reports success and the retry is suppressed forever.
"""

from __future__ import annotations

import json
import base64
import hashlib
from unittest.mock import MagicMock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from common.model_manifest import canonical_manifest_bytes

_PRIVATE_KEY = Ed25519PrivateKey.generate()
_PUBLIC_KEY = base64.b64encode(
    _PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
).decode()


def _make_service(edge_config, edge_db):
    """Build an InferenceService without touching MQTT/XGBoost."""
    from inference.main import InferenceService

    svc = InferenceService.__new__(InferenceService)
    svc.config = edge_config
    svc.db = edge_db
    svc.mqtt = MagicMock()
    svc.engine = MagicMock()
    svc._trusted_model_keys = {"test-key": _PUBLIC_KEY}
    svc._latest_telemetry = {}
    svc._running = False
    return svc


def _update_payload(models: list[dict]) -> dict:
    return {
        "action": "update",
        "target_version": "2.0.0",
        "channel": "stable",
        "models": models,
    }


def _model(name: str) -> dict:
    feature_contract = {
        "schema_version": 1,
        "model_name": name,
        "feature_names": ["f1"],
        "feature_count": 1,
        "missing_value_policy": "zero",
        "output_kind": "regression",
        "class_labels": [],
        "calibration_temperature": 1.0,
        "absolute_tolerance": 1e-5,
        "relative_tolerance": 1e-4,
    }
    contract_bytes = json.dumps(
        feature_contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    manifest = {
        "update_id": f"update-{name}",
        "model_name": name,
        "version": "2.0.0",
        "artifact_uri": f"s3://models/{name}.ubj",
        "path": f"/tmp/{name}.json",
        "sha256": "a" * 64,
        "size_bytes": 100,
        "target_nodes": ["*"],
        "rollout_pct": 100,
        "strategy": "stable",
        "timestamp": 1,
        "expires_at": "2099-01-01T00:00:00Z",
        "key_id": "test-key",
        "feature_contract": feature_contract,
        "golden_vectors": [{"inputs": {"f1": 1.0}, "expected": [2.0]}],
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
    }
    manifest["signature"] = base64.b64encode(
        _PRIVATE_KEY.sign(canonical_manifest_bytes(manifest))
    ).decode()
    return manifest


def _ota_status_payloads(edge_db) -> list[dict]:
    return [
        item["payload"]
        for item in edge_db.get_queue_items(priority=1)
        if item["payload"].get("type") == "ota_status"
    ]


def test_all_models_accepted_stamps_version(edge_config, edge_db):
    svc = _make_service(edge_config, edge_db)
    edge_db.set_config("firmware_version", "1.0.0")
    svc.engine.check_model_update.return_value = {"status": "accepted"}

    svc._on_model_update(_update_payload([_model("battery_soh")]))

    assert edge_db.get_config("firmware_version") == "2.0.0"
    statuses = _ota_status_payloads(edge_db)
    assert len(statuses) == 1
    assert statuses[0]["status"] == "success"
    assert statuses[0]["models_accepted"] == 1


def test_rejected_model_does_not_stamp_version(edge_config, edge_db):
    svc = _make_service(edge_config, edge_db)
    edge_db.set_config("firmware_version", "1.0.0")
    svc.engine.check_model_update.return_value = {"status": "rejected_sha256"}

    svc._on_model_update(_update_payload([_model("battery_soh")]))

    # Node must still report 1.0.0 so the cloud re-offers the update
    assert edge_db.get_config("firmware_version") == "1.0.0"
    statuses = _ota_status_payloads(edge_db)
    assert len(statuses) == 1
    assert statuses[0]["status"] == "no_models_accepted"
    assert statuses[0]["models_accepted"] == 0


def test_partial_acceptance_does_not_stamp_version(edge_config, edge_db):
    svc = _make_service(edge_config, edge_db)
    edge_db.set_config("firmware_version", "1.0.0")
    svc.engine.check_model_update.side_effect = [
        {"status": "accepted"},
        {"status": "rejected_accuracy"},
    ]

    svc._on_model_update(
        _update_payload([_model("battery_soh"), _model("load_forecast")])
    )

    assert edge_db.get_config("firmware_version") == "1.0.0"
    statuses = _ota_status_payloads(edge_db)
    assert len(statuses) == 1
    assert statuses[0]["status"] == "partial"
    assert statuses[0]["models_accepted"] == 1
    assert statuses[0]["models_total"] == 2


def test_manifestless_offer_stamps_version(edge_config, edge_db):
    """The wired cloud path (check_for_ota → /updates/pending) ships NO model
    manifest — such offers are version-stamp-only and MUST stamp, or every
    OTA offer livelocks (cloud re-offers the same version forever)."""
    svc = _make_service(edge_config, edge_db)
    edge_db.set_config("firmware_version", "1.0.0")

    svc._on_model_update(
        {"action": "update", "target_version": "2.0.0", "channel": "stable"}
    )

    assert edge_db.get_config("firmware_version") == "2.0.0"
    statuses = _ota_status_payloads(edge_db)
    assert len(statuses) == 1
    assert statuses[0]["status"] == "success"
    assert statuses[0]["models_total"] == 0


def test_empty_manifest_list_stamps_version(edge_config, edge_db):
    """Explicit models=[] is likewise a version-only offer, not a failure."""
    svc = _make_service(edge_config, edge_db)
    edge_db.set_config("firmware_version", "1.0.0")

    svc._on_model_update(_update_payload([]))

    assert edge_db.get_config("firmware_version") == "2.0.0"
    statuses = _ota_status_payloads(edge_db)
    assert statuses[0]["status"] == "success"


def test_missing_fields_model_counts_as_not_accepted(edge_config, edge_db):
    svc = _make_service(edge_config, edge_db)
    edge_db.set_config("firmware_version", "1.0.0")

    model = _model("battery_soh")
    del model["path"]
    svc._on_model_update(_update_payload([model]))

    assert edge_db.get_config("firmware_version") == "1.0.0"
    svc.engine.check_model_update.assert_not_called()
    statuses = _ota_status_payloads(edge_db)
    assert statuses[0]["model_results"] == {"battery_soh": "skipped_missing_fields"}


def test_model_only_update_never_stamps_firmware(edge_config, edge_db):
    svc = _make_service(edge_config, edge_db)
    edge_db.set_config("firmware_version", "1.0.0")
    svc.engine.check_model_update.return_value = {"status": "accepted"}
    payload = _update_payload([_model("battery_soh")])
    payload["firmware_update"] = False

    svc._on_model_update(payload)

    assert edge_db.get_config("firmware_version") == "1.0.0"


def test_ota_status_payload_roundtrips_through_queue(edge_config, edge_db):
    """The queue stores gzip'd JSON — confirm the payload survives intact."""
    svc = _make_service(edge_config, edge_db)
    svc.engine.check_model_update.return_value = {"status": "accepted"}

    svc._on_model_update(_update_payload([_model("battery_soh")]))

    payload = _ota_status_payloads(edge_db)[0]
    assert json.loads(json.dumps(payload)) == payload
    assert payload["device_id"] == "test-edge-001"
    assert payload["target_version"] == "2.0.0"
