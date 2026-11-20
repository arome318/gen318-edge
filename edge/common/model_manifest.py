"""Verify cloud-signed model manifests without importing the backend."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

SIGNED_FIELDS = (
    "update_id",
    "model_name",
    "version",
    "artifact_uri",
    "sha256",
    "size_bytes",
    "target_nodes",
    "rollout_pct",
    "strategy",
    "timestamp",
    "expires_at",
    "key_id",
    "feature_contract",
    "golden_vectors",
    "contract_sha256",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    signed = {field: manifest.get(field) for field in SIGNED_FIELDS}
    return json.dumps(
        signed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def parse_trusted_keys(encoded_json: str) -> dict[str, str]:
    if not encoded_json.strip():
        return {}
    try:
        value = json.loads(encoded_json)
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(encoded)
        for key, encoded in value.items()
        if isinstance(key, str) and isinstance(encoded, str)
    }


def verify_manifest(
    manifest: Mapping[str, Any],
    trusted_keys: Mapping[str, str],
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    for field in ("update_id", "model_name", "version", "artifact_uri", "strategy"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            return False, f"invalid_{field}"
    if not _SHA256_RE.fullmatch(str(manifest.get("sha256", ""))):
        return False, "invalid_sha256"
    if not isinstance(manifest.get("size_bytes"), int) or manifest["size_bytes"] <= 0:
        return False, "invalid_size"
    targets = manifest.get("target_nodes")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(target, str) and target for target in targets)
    ):
        return False, "invalid_targets"
    rollout_pct = manifest.get("rollout_pct")
    if not isinstance(rollout_pct, int) or not 1 <= rollout_pct <= 100:
        return False, "invalid_rollout_pct"
    if not isinstance(manifest.get("timestamp"), int) or manifest["timestamp"] <= 0:
        return False, "invalid_timestamp"

    contract = manifest.get("feature_contract")
    vectors = manifest.get("golden_vectors")
    if not isinstance(contract, dict) or not contract:
        return False, "missing_feature_contract"
    feature_names = contract.get("feature_names")
    if (
        contract.get("schema_version") != 1
        or contract.get("model_name") != manifest.get("model_name")
        or not isinstance(feature_names, list)
        or not feature_names
        or len(set(feature_names)) != len(feature_names)
        or contract.get("feature_count") != len(feature_names)
        or contract.get("missing_value_policy") != "zero"
        or contract.get("output_kind") not in {"regression", "classification"}
    ):
        return False, "invalid_feature_contract"
    contract_bytes = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    if manifest.get("contract_sha256") != hashlib.sha256(contract_bytes).hexdigest():
        return False, "feature_contract_hash_mismatch"
    if not isinstance(vectors, list) or not vectors:
        return False, "missing_golden_vectors"
    for vector in vectors:
        if not isinstance(vector, dict):
            return False, "invalid_golden_vector"
        inputs = vector.get("inputs")
        expected = vector.get("expected")
        if (
            not isinstance(inputs, dict)
            or set(inputs) != set(feature_names)
            or not all(
                isinstance(value, int | float) and math.isfinite(value)
                for value in inputs.values()
            )
            or not isinstance(expected, list)
            or not expected
            or not all(
                isinstance(value, int | float) and math.isfinite(value)
                for value in expected
            )
        ):
            return False, "invalid_golden_vector"

    key_id = manifest.get("key_id")
    signature_encoded = manifest.get("signature")
    if not isinstance(key_id, str) or key_id not in trusted_keys:
        return False, "untrusted_key"
    if not isinstance(signature_encoded, str):
        return False, "missing_signature"

    expires_at = manifest.get("expires_at")
    if not isinstance(expires_at, str):
        return False, "missing_expiry"
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return False, "invalid_expiry"
    if expiry.tzinfo is None:
        return False, "invalid_expiry"
    if expiry <= (now or datetime.now(UTC)):
        return False, "expired"

    try:
        public_raw = base64.b64decode(trusted_keys[key_id], validate=True)
        signature = base64.b64decode(signature_encoded, validate=True)
        if len(public_raw) != 32 or len(signature) != 64:
            return False, "invalid_key_or_signature_length"
        Ed25519PublicKey.from_public_bytes(public_raw).verify(
            signature,
            canonical_manifest_bytes(manifest),
        )
    except (ValueError, TypeError, InvalidSignature):
        return False, "invalid_signature"
    return True, "verified"
