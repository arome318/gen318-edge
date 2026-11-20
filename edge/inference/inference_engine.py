"""Edge ML inference engine — loads and runs XGBoost models on-device (Doc 14 §8.2).

Supports battery health, battery RUL, inverter anomaly, demand forecast
(all XGBoost 3.2.0 native), plus a rule-based dispatch advisory.
SHA-256 verification before loading; accuracy-based auto-rollback.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from common.database import EdgeDatabase

logger = structlog.get_logger()

# Model definitions mirror the cloud training constants exactly. A signed
# contract must match these names/order before an artifact can activate.
MODEL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "battery_health": {
        "file": "battery_health.ubj",
        "features": [
            "cumulative_ah_throughput",
            "estimated_cycles",
            "temp_stress_degree_hours",
            "avg_temperature_c",
            "max_temperature_c",
            "avg_discharge_depth_pct",
            "max_discharge_depth_pct",
            "deep_discharge_count",
            "cell_voltage_spread_avg_v",
            "cell_voltage_spread_max_v",
            "voltage_sag_v",
            "chemistry_calendar_factor",
            "chemistry_dod_sensitivity",
            "chemistry_cycle_life_nominal",
            "age_days",
            "avg_charge_rate_a",
            "max_charge_rate_a",
            "ica_peak_height",
            "ica_peak_voltage",
            "ica_area",
            "dva_plateau_length",
            "dva_slope_mean",
            "relaxation_voltage_std",
            "relaxation_voltage_drift",
            "sei_growth_proxy",
        ],
        "output": "soh_pct",
    },
    "battery_rul": {
        "file": "battery_rul.ubj",
        "features": [
            "calendar_aging_factor",
            "sei_growth_rate",
            "cycle_count_weighted",
            "avg_temperature_c",
            "max_temperature_c",
            "avg_depth_of_discharge_pct",
            "deep_discharge_count",
            "current_soh_pct",
            "soh_decline_rate_30d",
            "cell_voltage_spread_avg_v",
            "voltage_sag_v",
            "age_days",
            "chemistry_cycle_life_nominal",
        ],
        "output": "rul_days",
    },
    "inverter_fault_coarse": {
        "file": "inverter_fault_coarse.ubj",
        "features": [
            "efficiency_1h_mean",
            "efficiency_1h_std",
            "efficiency_24h_mean",
            "efficiency_24h_std",
            "efficiency_7d_mean",
            "efficiency_7d_trend",
            "temperature_1h_mean",
            "temperature_1h_max",
            "temperature_24h_mean",
            "temperature_7d_trend",
            "temp_vs_load_correlation",
            "ac_output_voltage_1h_std",
            "ac_output_voltage_24h_range",
            "voltage_thd_estimate",
            "ac_output_current_1h_std",
            "current_thd_estimate",
            "frequency_1h_std",
            "frequency_24h_range",
            "error_code_frequency_24h",
            "unique_error_codes_7d",
            "power_factor_1h_mean",
            "power_factor_1h_std",
            "load_ratio_1h_mean",
            "overload_minutes_24h",
            "comm_timeout_count_24h",
            "comm_gap_max_seconds_24h",
        ],
        "output": "fault_group",
    },
    "inverter_fault_full": {
        "file": "inverter_fault_full.ubj",
        "features": [],  # filled from the coarse definition immediately below
        "output": "fault_type",
    },
    "demand_forecast": {
        "file": "demand_forecast.json",
        "features": [
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "load_1h",
            "load_2h",
            "load_24h",
            "temperature_c",
            "is_holiday",
            "is_market_day",
        ],
        "output": "load_24h_forecast",
    },
}
MODEL_DEFINITIONS["inverter_fault_full"]["features"] = list(
    MODEL_DEFINITIONS["inverter_fault_coarse"]["features"]
)

# Dispatch advisory lookup table (rule-based, no ML)
DISPATCH_TABLE: dict[str, dict[str, Any]] = {
    "high_solar_low_load": {
        "action": "charge_battery",
        "priority": "normal",
        "description": "Surplus solar — charge battery to max SoC",
    },
    "low_solar_high_load_battery_ok": {
        "action": "discharge_battery",
        "priority": "normal",
        "description": "Use battery to cover load deficit",
    },
    "low_solar_high_load_battery_low": {
        "action": "start_diesel",
        "priority": "high",
        "description": "Battery low — start diesel genset",
    },
    "peak_hours_battery_ok": {
        "action": "discharge_battery",
        "priority": "normal",
        "description": "Evening peak — serve from battery",
    },
    "night_low_load": {
        "action": "idle",
        "priority": "low",
        "description": "Low overnight load — maintain SoC",
    },
}


class InferenceEngine:
    """On-device ML inference engine using XGBoost native format."""

    def __init__(self, model_path: str, db: EdgeDatabase) -> None:
        self._model_path = Path(model_path)
        self._db = db
        self._models: dict[str, Any] = {}
        self._contracts: dict[str, dict[str, Any]] = {}
        self._xgb_available = False

        try:
            import xgboost as xgb

            self._xgb = xgb
            self._xgb_available = True
        except ImportError:
            logger.warning(
                "xgboost_not_available",
                msg="XGBoost not installed — inference will return defaults",
            )

    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def load_models(self) -> dict[str, bool]:
        """Load all available XGBoost models from disk, verify SHA-256."""
        results: dict[str, bool] = {}

        for model_name, definition in MODEL_DEFINITIONS.items():
            model_file = self._model_path / definition["file"]
            if not model_file.exists():
                logger.info(
                    "model_file_not_found",
                    model=model_name,
                    path=str(model_file),
                )
                results[model_name] = False
                continue

            if not self._xgb_available:
                results[model_name] = False
                continue

            try:
                # Verify SHA-256 against stored state
                file_sha256 = self._compute_sha256(model_file)
                stored_state = self._db.get_model_state()
                stored = next(
                    (s for s in stored_state if s["model_name"] == model_name),
                    None,
                )

                if stored and stored["sha256"] != file_sha256:
                    logger.warning(
                        "model_sha256_mismatch",
                        model=model_name,
                        expected=stored["sha256"],
                        actual=file_sha256,
                    )
                    results[model_name] = False
                    continue

                # Load the XGBoost model
                booster = self._xgb.Booster()
                booster.load_model(str(model_file))
                self._models[model_name] = booster
                if stored and isinstance(stored.get("feature_contract"), dict):
                    self._contracts[model_name] = stored["feature_contract"]

                # Update model state in DB (hash matches, safe to record)
                self._db.update_model_state(
                    model_name=model_name,
                    version=stored["version"] if stored else "1.0.0",
                    sha256=file_sha256,
                    feature_contract=(
                        stored.get("feature_contract") if stored else None
                    ),
                )

                results[model_name] = True
                logger.info(
                    "model_loaded",
                    model=model_name,
                    sha256=file_sha256[:16] + "...",
                )

            except Exception:
                logger.exception("model_load_error", model=model_name)
                results[model_name] = False

        return results

    def predict(self, model_name: str, features: dict[str, float]) -> dict[str, Any]:
        """Run inference on a loaded model."""
        if model_name not in self._models:
            return {
                "model": model_name,
                "status": "model_not_loaded",
                "prediction": None,
            }

        definition = MODEL_DEFINITIONS[model_name]
        feature_names = definition["features"]

        # Build feature vector in correct order
        feature_values = [features.get(f, 0.0) for f in feature_names]
        dmatrix = self._xgb.DMatrix(
            np.array([feature_values]),
            feature_names=feature_names,
        )

        start = time.monotonic()
        prediction = self._models[model_name].predict(dmatrix)
        inference_ms = (time.monotonic() - start) * 1000

        raw_output = np.asarray(prediction[0], dtype=np.float64).reshape(-1)
        contract = self._contracts.get(model_name, {})
        if contract.get("output_kind") == "classification" or len(raw_output) > 1:
            temperature = float(contract.get("calibration_temperature", 1.0))
            logits = np.log(np.clip(raw_output, 1e-12, 1.0)) / max(temperature, 0.05)
            probabilities = np.exp(logits - np.max(logits))
            probabilities /= probabilities.sum()
            class_index = int(np.argmax(probabilities))
            labels = contract.get("class_labels", [])
            result_value: float | int = class_index
            output_details: dict[str, Any] = {
                "probabilities": [round(float(value), 8) for value in probabilities],
                "confidence": round(float(probabilities[class_index]), 8),
                "class_label": labels[class_index]
                if class_index < len(labels)
                else None,
            }
        else:
            result_value = float(raw_output[0])
            output_details = {}

        logger.debug(
            "inference_complete",
            model=model_name,
            inference_ms=round(inference_ms, 2),
            result=result_value,
        )

        return {
            "model": model_name,
            "status": "ok",
            "prediction": result_value,
            "output_name": definition["output"],
            "inference_ms": round(inference_ms, 2),
            "timestamp": int(time.time() * 1000),
            "feature_contract_schema": contract.get("schema_version"),
            **output_details,
        }

    def predict_battery_health(self, features: dict[str, float]) -> dict[str, Any]:
        """Battery state-of-health estimation."""
        return self.predict("battery_health", features)

    def predict_battery_rul(self, features: dict[str, float]) -> dict[str, Any]:
        """Battery remaining useful life prediction."""
        return self.predict("battery_rul", features)

    def predict_inverter_fault(
        self, features: dict[str, float], taxonomy: str = "coarse"
    ) -> dict[str, Any]:
        """Inverter coarse/full fault classification."""
        return self.predict(f"inverter_fault_{taxonomy}", features)

    def predict_demand_forecast(self, features: dict[str, float]) -> dict[str, Any]:
        """Next-24h load prediction."""
        return self.predict("demand_forecast", features)

    def run_dispatch_advisory(self, state: dict[str, float]) -> dict[str, Any]:
        """Rule-based dispatch advisory (no ML, <5ms)."""
        solar_w = state.get("solar_pv_power_w", 0)
        load_w = state.get("ac_consumption_w", 0)
        soc_pct = state.get("battery_soc_pct", 50)
        hour = state.get("hour", 12)

        surplus = solar_w - load_w

        if surplus > load_w * 0.3 and soc_pct < 90:
            scenario = "high_solar_low_load"
        elif surplus < 0 and soc_pct > 30:
            scenario = "low_solar_high_load_battery_ok"
        elif surplus < 0 and soc_pct <= 30:
            scenario = "low_solar_high_load_battery_low"
        elif 18 <= hour <= 22 and soc_pct > 30:
            scenario = "peak_hours_battery_ok"
        else:
            scenario = "night_low_load"

        advisory = DISPATCH_TABLE[scenario]
        return {
            "model": "dispatch_advisory",
            "status": "ok",
            "scenario": scenario,
            "timestamp": int(time.time() * 1000),
            **advisory,
        }

    def check_model_update(
        self,
        model_name: str,
        new_model_path: Path,
        expected_sha256: str,
        feature_contract: dict[str, Any],
        golden_vectors: list[dict[str, Any]],
        model_version: str = "unknown",
    ) -> dict[str, Any]:
        """Verify and load a new model with mandatory parity validation.

        Returns status dict with 'accepted', 'rejected_sha256',
        or 'rejected_accuracy'.
        """
        if model_name not in MODEL_DEFINITIONS:
            return {"model": model_name, "status": "rejected_unknown_model"}
        expected_features = MODEL_DEFINITIONS[model_name]["features"]
        if (
            feature_contract.get("schema_version") != 1
            or feature_contract.get("model_name") != model_name
            or feature_contract.get("feature_names") != expected_features
            or feature_contract.get("feature_count") != len(expected_features)
            or feature_contract.get("missing_value_policy") != "zero"
        ):
            return {"model": model_name, "status": "rejected_feature_contract"}
        if not golden_vectors:
            return {"model": model_name, "status": "rejected_missing_golden_vectors"}

        # Step 1: SHA-256 verification
        actual_sha256 = self._compute_sha256(new_model_path)
        if actual_sha256 != expected_sha256:
            logger.warning(
                "model_update_sha256_mismatch",
                model=model_name,
                expected=expected_sha256,
                actual=actual_sha256,
            )
            return {
                "model": model_name,
                "status": "rejected_sha256",
                "expected": expected_sha256,
                "actual": actual_sha256,
            }

        if not self._xgb_available:
            return {"model": model_name, "status": "rejected_no_xgboost"}

        # Step 2: Load candidate model
        try:
            candidate = self._xgb.Booster()
            candidate.load_model(str(new_model_path))
        except Exception:
            logger.exception("model_update_load_error", model=model_name)
            return {"model": model_name, "status": "rejected_load_error"}

        # Step 3: exact cloud-produced golden-vector parity. Empty validation
        # can never fall through to activation.
        absolute_tolerance = float(feature_contract.get("absolute_tolerance", 1e-5))
        relative_tolerance = float(feature_contract.get("relative_tolerance", 1e-4))
        for vector_index, vector in enumerate(golden_vectors):
            inputs = vector.get("inputs", {})
            expected = np.asarray(vector.get("expected", []), dtype=np.float64).reshape(
                -1
            )
            if set(inputs) != set(expected_features) or not len(expected):
                return {
                    "model": model_name,
                    "status": "rejected_golden_vector_contract",
                }
            dmatrix = self._xgb.DMatrix(
                np.asarray([[float(inputs[name]) for name in expected_features]]),
                feature_names=expected_features,
            )
            actual = np.asarray(
                candidate.predict(dmatrix)[0], dtype=np.float64
            ).reshape(-1)
            if feature_contract.get("output_kind") == "classification":
                temperature = float(
                    feature_contract.get("calibration_temperature", 1.0)
                )
                logits = np.log(np.clip(actual, 1e-12, 1.0)) / max(temperature, 0.05)
                actual = np.exp(logits - np.max(logits))
                actual /= actual.sum()
            if not np.allclose(
                actual,
                expected,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                equal_nan=False,
            ):
                logger.warning(
                    "model_update_golden_vector_mismatch",
                    model=model_name,
                    vector_index=vector_index,
                    expected=expected.tolist(),
                    actual=actual.tolist(),
                )
                return {
                    "model": model_name,
                    "status": "rejected_golden_vector_mismatch",
                    "vector_index": vector_index,
                }

        # Step 4: Accept — replace current model
        target_path = self._model_path / MODEL_DEFINITIONS[model_name]["file"]
        backup_path = target_path.with_suffix(target_path.suffix + ".bak")
        temporary_path = target_path.with_suffix(target_path.suffix + ".new")
        had_target = target_path.exists()
        previous_model = self._models.get(model_name)

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(new_model_path, temporary_path)
            with temporary_path.open("rb") as staged:
                os.fsync(staged.fileno())
            if had_target:
                shutil.copy2(target_path, backup_path)
            os.replace(temporary_path, target_path)
            self._models[model_name] = candidate
            self._db.update_model_state(
                model_name=model_name,
                version=model_version,
                sha256=actual_sha256,
                feature_contract=feature_contract,
            )
            self._contracts[model_name] = feature_contract
        except Exception:
            logger.exception("model_update_activation_failed", model=model_name)
            temporary_path.unlink(missing_ok=True)
            if had_target and backup_path.exists():
                os.replace(backup_path, target_path)
            elif not had_target:
                target_path.unlink(missing_ok=True)
            if previous_model is None:
                self._models.pop(model_name, None)
            else:
                self._models[model_name] = previous_model
            return {"model": model_name, "status": "rejected_activation_error"}

        logger.info(
            "model_update_accepted",
            model=model_name,
            sha256=actual_sha256[:16] + "...",
        )
        return {"model": model_name, "status": "accepted", "sha256": actual_sha256}
