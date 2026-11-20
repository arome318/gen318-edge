"""Edge inference service — subscribes MQTT telemetry, runs on-device ML (Doc 14).

Runs as the gen318-inference container. Periodically runs XGBoost models
on buffered telemetry and publishes results to MQTT + SQLite.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any

import structlog
from common.config import EdgeConfig, load_config
from common.database import EdgeDatabase
from common.model_manifest import parse_trusted_keys, verify_manifest
from common.mqtt_client import (
    TOPIC_INFERENCE_RESULTS,
    TOPIC_SITELEARN,
    TOPIC_MODEL_UPDATES,
    TOPIC_TELEMETRY,
    MQTTClient,
)

from inference.inference_engine import InferenceEngine
from inference.sitelearn import SiteLearnConfig, SiteLearner
from inference.features import (
    build_battery_health_features,
    build_battery_rul_features,
    build_inverter_features,
)

logger = structlog.get_logger()


class InferenceService:
    """Runs ML inference on edge telemetry at regular intervals."""

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self.db = EdgeDatabase(config.sqlite_path, config.sqlite_encryption_key)
        self.mqtt = MQTTClient(
            host=config.mqtt_host,
            port=config.mqtt_port,
            client_id="gen318-inference",
        )
        self.engine = InferenceEngine(config.model_path, self.db)
        self._trusted_model_keys = parse_trusted_keys(
            config.model_trusted_public_keys_json
        )
        self._latest_telemetry: dict[str, dict[str, float]] = {}
        self._running = False
        self.sitelearn = SiteLearner(
            self.db,
            config.model_path,
            SiteLearnConfig(utc_offset_hours=float(os.environ.get("SITE_UTC_OFFSET_HOURS", "1"))),
        )
        self._sitelearn_interval = float(os.environ.get("SITELEARN_INTERVAL_SECONDS", "900"))
        self._sitelearn_last = 0.0

    def _run_site_learning(self) -> None:
        """Train due on-site models and publish current findings (warnings only)."""
        now = time.time()
        if now - self._sitelearn_last < self._sitelearn_interval:
            return
        self._sitelearn_last = now
        try:
            findings = self.sitelearn.step(now)
        except Exception:
            logger.exception("site_learning_failed")
            return
        self.mqtt.publish(TOPIC_SITELEARN, {"at": now, "findings": findings})
        logger.info("site_learning_cycle", findings=[f["code"] for f in findings])

    def _on_model_update(self, payload: dict[str, Any]) -> None:
        """Handle model update command from sync engine via MQTT.

        Uses check_model_update() for OTA updates so that SHA-256
        verification and accuracy-based rollback are enforced.
        """
        action = payload.get("action", "")
        target_version = payload.get("target_version", "")

        models = payload.get("models", [])
        if action != "update" or (not target_version and not models):
            return

        logger.info(
            "model_update_received",
            target_version=target_version,
            channel=payload.get("channel", "stable"),
        )

        update_results: dict[str, str] = {}

        for model_info in models:
            model_name = model_info.get("model_name", "")
            new_path = model_info.get("path", "")
            expected_sha256 = model_info.get("sha256", "")
            feature_contract = model_info.get("feature_contract", {})
            golden_vectors = model_info.get("golden_vectors", [])

            verified, verification_reason = verify_manifest(
                model_info, self._trusted_model_keys
            )
            if not verified:
                update_results[model_name or "unknown"] = verification_reason
                logger.warning(
                    "model_manifest_rejected_at_activation",
                    model=model_name,
                    reason=verification_reason,
                )
                continue
            if not model_name or not new_path or not expected_sha256:
                logger.warning(
                    "model_update_missing_fields",
                    model=model_name,
                )
                update_results[model_name] = "skipped_missing_fields"
                continue

            result = self.engine.check_model_update(
                model_name=model_name,
                new_model_path=Path(new_path),
                expected_sha256=expected_sha256,
                feature_contract=feature_contract,
                golden_vectors=golden_vectors,
                model_version=str(model_info.get("version", "unknown")),
            )
            update_results[model_name] = result["status"]

        accepted = sum(1 for s in update_results.values() if s == "accepted")
        # A manifest with models must FULLY verify to stamp — stamping a
        # partial/failed firmware bundle would self-report success and suppress
        # retry forever. A firmware offer without models remains a legitimate
        # version-only update; model-only offers never stamp firmware below.
        fully_applied = accepted == len(update_results)  # vacuously true if empty

        if fully_applied and payload.get("firmware_update", True):
            self.db.set_config("firmware_version", target_version)

        if fully_applied:
            status = "success"
        elif accepted > 0:
            status = "partial"
        else:
            status = "no_models_accepted"

        # Report status via sync queue (P1)
        now_ms = int(time.time() * 1000)
        self.db.enqueue_sync(
            priority=1,
            payload={
                "type": "ota_status",
                "device_id": self.config.device_id,
                "target_version": target_version,
                "models_accepted": accepted,
                "models_total": len(update_results),
                "model_results": update_results,
                "status": status,
                "timestamp": now_ms,
            },
        )

        logger.info(
            "model_update_processed",
            version=target_version,
            status=status,
            models_accepted=accepted,
            model_results=update_results,
        )

    def _on_telemetry(self, payload: dict[str, Any]) -> None:
        """Callback for incoming telemetry — buffer latest values per asset."""
        asset_id = payload.get("asset_id", "")
        metric = payload.get("metric_name", "")
        value = payload.get("value", 0.0)

        if asset_id not in self._latest_telemetry:
            self._latest_telemetry[asset_id] = {}
        self._latest_telemetry[asset_id][metric] = value

    async def _run_inference_cycle(self) -> None:
        """Run all inference models on buffered telemetry."""
        if not self._latest_telemetry:
            logger.debug("inference_skip_no_data")
            return

        now_ms = int(time.time() * 1000)
        results: list[dict[str, Any]] = []

        for asset_id, metrics in self._latest_telemetry.items():
            # Battery health
            if "battery.soc_pct" in metrics or "battery.voltage_v" in metrics:
                battery_features = build_battery_health_features(self.db, asset_id)
                result = self.engine.predict_battery_health(battery_features)
                result["asset_id"] = asset_id
                results.append(result)

                # Battery RUL
                rul_features = build_battery_rul_features(
                    self.db, asset_id, battery_features
                )
                rul_result = self.engine.predict_battery_rul(rul_features)
                rul_result["asset_id"] = asset_id
                results.append(rul_result)

            # Inverter anomaly
            if "inverter.ac_output_power_w" in metrics:
                rated_power = float(
                    self.db.get_config(f"asset:{asset_id}:rated_power_w") or 0.0
                )
                inverter_features = build_inverter_features(
                    self.db, asset_id, rated_power_w=rated_power
                )
                inv_result = self.engine.predict_inverter_fault(
                    inverter_features, taxonomy="full"
                )
                if inv_result["status"] == "model_not_loaded":
                    inv_result = self.engine.predict_inverter_fault(
                        inverter_features, taxonomy="coarse"
                    )
                inv_result["asset_id"] = asset_id
                results.append(inv_result)

            # Dispatch advisory (rule-based)
            dispatch_state = {
                "solar_pv_power_w": metrics.get("solar.pv_power_w", 0),
                "ac_consumption_w": metrics.get("system.ac_consumption_w", 0),
                "battery_soc_pct": metrics.get("battery.soc_pct", 50),
                "hour": time.localtime().tm_hour,
            }
            dispatch_result = self.engine.run_dispatch_advisory(dispatch_state)
            dispatch_result["asset_id"] = asset_id
            results.append(dispatch_result)

        # Publish results and store events
        for result in results:
            if result.get("status") == "ok":
                self.mqtt.publish(TOPIC_INFERENCE_RESULTS, result)

                # Store as event in SQLite
                self.db.insert_event(
                    event_id=str(uuid.uuid4()),
                    event_type=f"inference.{result['model']}",
                    severity="info",
                    payload=result,
                    timestamp=now_ms,
                )

                # Enqueue for cloud sync as P1 (high priority)
                self.db.enqueue_sync(priority=1, payload=result)

        logger.info(
            "inference_cycle_complete",
            assets=len(self._latest_telemetry),
            results=len(results),
        )

    async def run(self) -> None:
        """Main inference loop."""
        self.db.connect()
        self.db.init_schema()
        self.mqtt.connect()

        # Load models
        load_results = self.engine.load_models()
        loaded = sum(1 for v in load_results.values() if v)
        logger.info(
            "inference_started",
            models_loaded=loaded,
            models_total=len(load_results),
            interval_s=self.config.inference_interval_seconds,
        )

        # Subscribe to telemetry for real-time feature buffering
        self.mqtt.subscribe(TOPIC_TELEMETRY, self._on_telemetry)

        # Subscribe to model update commands from sync engine
        self.mqtt.subscribe(TOPIC_MODEL_UPDATES, self._on_model_update)

        self._running = True
        while self._running:
            try:
                await self._run_inference_cycle()
            except Exception:
                logger.exception("inference_cycle_error")
            await asyncio.to_thread(self._run_site_learning)

            await asyncio.sleep(self.config.inference_interval_seconds)

    def stop(self) -> None:
        self._running = False
        self.mqtt.disconnect()
        self.db.close()
        logger.info("inference_stopped")


async def main() -> None:
    config = load_config()
    service = InferenceService(config)
    try:
        await service.run()
    except KeyboardInterrupt:
        service.stop()


if __name__ == "__main__":
    asyncio.run(main())
