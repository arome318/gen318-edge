"""Fail-closed edge activation tests for signed feature contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from inference.inference_engine import InferenceEngine, MODEL_DEFINITIONS


class _DMatrix:
    def __init__(self, values, feature_names) -> None:
        self.values = np.asarray(values, dtype=np.float64)
        self.feature_names = feature_names


class _Booster:
    def load_model(self, _path: str) -> None:
        return None

    def predict(self, matrix: _DMatrix) -> np.ndarray:
        return np.asarray([float(np.sum(matrix.values))])


class _XGBoost:
    Booster = _Booster
    DMatrix = _DMatrix


def _engine(tmp_path: Path, edge_db) -> InferenceEngine:
    engine = InferenceEngine(str(tmp_path), edge_db)
    engine._xgb_available = True
    engine._xgb = _XGBoost
    return engine


def _contract(model_name: str) -> dict:
    features = MODEL_DEFINITIONS[model_name]["features"]
    return {
        "schema_version": 1,
        "model_name": model_name,
        "feature_names": features,
        "feature_count": len(features),
        "missing_value_policy": "zero",
        "output_kind": "regression",
        "class_labels": [],
        "calibration_temperature": 1.0,
        "absolute_tolerance": 1e-8,
        "relative_tolerance": 1e-8,
    }


def _vector(model_name: str, expected: float | None = None) -> dict:
    inputs = {
        feature: float(index + 1)
        for index, feature in enumerate(MODEL_DEFINITIONS[model_name]["features"])
    }
    return {
        "inputs": inputs,
        "expected": [sum(inputs.values()) if expected is None else expected],
    }


def test_matching_contract_and_golden_vector_activate_atomically(
    tmp_path, edge_db
) -> None:
    artifact = tmp_path / "candidate.ubj"
    artifact.write_bytes(b"candidate")
    engine = _engine(tmp_path, edge_db)
    model_name = "battery_rul"

    result = engine.check_model_update(
        model_name,
        artifact,
        hashlib.sha256(artifact.read_bytes()).hexdigest(),
        _contract(model_name),
        [_vector(model_name)],
        model_version="7",
    )

    assert result["status"] == "accepted"
    state = edge_db.get_model_state()[0]
    assert state["version"] == "7"
    assert (
        state["feature_contract"]["feature_names"]
        == MODEL_DEFINITIONS[model_name]["features"]
    )


def test_missing_golden_vectors_can_never_activate(tmp_path, edge_db) -> None:
    artifact = tmp_path / "candidate.ubj"
    artifact.write_bytes(b"candidate")
    model_name = "battery_rul"
    result = _engine(tmp_path, edge_db).check_model_update(
        model_name,
        artifact,
        hashlib.sha256(artifact.read_bytes()).hexdigest(),
        _contract(model_name),
        [],
    )
    assert result["status"] == "rejected_missing_golden_vectors"


def test_cloud_edge_feature_order_mismatch_is_rejected(tmp_path, edge_db) -> None:
    artifact = tmp_path / "candidate.ubj"
    artifact.write_bytes(b"candidate")
    model_name = "battery_rul"
    contract = _contract(model_name)
    contract["feature_names"] = list(reversed(contract["feature_names"]))
    result = _engine(tmp_path, edge_db).check_model_update(
        model_name,
        artifact,
        hashlib.sha256(artifact.read_bytes()).hexdigest(),
        contract,
        [_vector(model_name)],
    )
    assert result["status"] == "rejected_feature_contract"


def test_golden_output_mismatch_preserves_current_model(tmp_path, edge_db) -> None:
    artifact = tmp_path / "candidate.ubj"
    artifact.write_bytes(b"candidate")
    model_name = "battery_rul"
    result = _engine(tmp_path, edge_db).check_model_update(
        model_name,
        artifact,
        hashlib.sha256(artifact.read_bytes()).hexdigest(),
        _contract(model_name),
        [_vector(model_name, expected=-999.0)],
    )
    assert result["status"] == "rejected_golden_vector_mismatch"
    assert edge_db.get_model_state() == []
