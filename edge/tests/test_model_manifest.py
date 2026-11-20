from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from common.model_manifest import canonical_manifest_bytes, verify_manifest


def _signed_manifest() -> tuple[dict, dict[str, str]]:
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    manifest = {
        "update_id": "0155b28e-ff99-4b1e-856a-9cc26ea590e4",
        "model_name": "battery_health",
        "version": "3",
        "artifact_uri": "s3://models/battery_health/3/model.ubj",
        "sha256": "a" * 64,
        "size_bytes": 128,
        "target_nodes": ["*"],
        "rollout_pct": 100,
        "strategy": "stable",
        "timestamp": 1,
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        "key_id": "test-key",
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


def test_valid_manifest_verifies() -> None:
    manifest, keys = _signed_manifest()
    assert verify_manifest(manifest, keys) == (True, "verified")


def test_signed_field_tampering_is_rejected() -> None:
    manifest, keys = _signed_manifest()
    manifest["sha256"] = "b" * 64
    assert verify_manifest(manifest, keys) == (False, "invalid_signature")


def test_feature_contract_tampering_is_rejected_before_signature() -> None:
    manifest, keys = _signed_manifest()
    manifest["feature_contract"]["feature_names"] = ["different"]
    assert verify_manifest(manifest, keys) == (False, "feature_contract_hash_mismatch")


def test_empty_golden_vectors_are_rejected() -> None:
    manifest, keys = _signed_manifest()
    manifest["golden_vectors"] = []
    assert verify_manifest(manifest, keys) == (False, "missing_golden_vectors")


def test_expired_manifest_is_rejected() -> None:
    manifest, keys = _signed_manifest()
    future = datetime.now(UTC) + timedelta(hours=2)
    assert verify_manifest(manifest, keys, now=future) == (False, "expired")


def test_ephemeral_download_url_is_not_part_of_signature() -> None:
    manifest, keys = _signed_manifest()
    manifest["download_url"] = "https://objects.example/signed?expires=1"
    manifest["path"] = "/models/staging/model.ubj"
    assert verify_manifest(manifest, keys) == (True, "verified")
