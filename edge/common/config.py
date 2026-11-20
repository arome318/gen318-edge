"""Edge node configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EdgeConfig:
    """Configuration for the Gen318 edge node stack."""

    # Device identity
    device_id: str
    cloud_api_url: str
    device_cert_path: str
    device_key_path: str
    ca_cert_path: str
    cert_pin_sha256: str

    # Storage
    sqlite_path: str
    sqlite_encryption_key: str
    storage_target_days: int = 30
    storage_expected_assets: int = 10
    storage_metrics_per_asset: int = 40
    storage_bytes_per_measurement: int = 160
    storage_reserve_bytes: int = 2 * 1024**3
    storage_warning_used_pct: float = 80.0
    storage_critical_used_pct: float = 90.0
    storage_emergency_used_pct: float = 96.0
    # Standalone boxes (no CLOUD_API_URL) keep this many days of readings.
    # The on-site models look back 30 days, so do not go below about 35.
    local_retention_days: int = 45

    # Adapters
    adapter_config_path: str = "/config/adapters.yaml"

    # Models
    model_path: str = "/models"
    model_trusted_public_keys_json: str = "{}"
    model_artifact_max_bytes: int = 50 * 1024 * 1024

    # Sync
    sync_interval_seconds: int = 60
    backfill_max_batch_mb: int = 5
    p1_max_items_on_2g: int = 10
    telemetry_batch_size: int = 1000
    telemetry_max_batches_per_cycle: int = 10
    telemetry_recent_window_hours: int = 24
    # Synced rows are kept this long. The on-site models look back 30 days,
    # so a syncing box must not delete its history after a few days.
    telemetry_retention_hours: int = 1080

    # Collection
    poll_interval_seconds: int = 60

    # Register probe (opt-in, read-only mapping evidence — never telemetry)
    register_probe_enabled: bool = False
    register_probe_host: str = ""
    register_probe_port: int = 502
    register_probe_ranges: str = "225:280-310,225:325-335"
    register_probe_interval_seconds: int = 3600

    # Inference
    inference_interval_seconds: int = 300

    # Watchdog
    check_interval: int = 60
    tamper_gpio_pin: int = 17
    monitored_containers: list[str] = field(
        default_factory=lambda: [
            "gen318-mqtt",
            "gen318-collector",
            "gen318-inference",
            "gen318-sync",
            "gen318-api",
        ]
    )

    # MQTT
    mqtt_host: str = "mosquitto"
    mqtt_port: int = 1883


def load_config() -> EdgeConfig:
    """Load edge configuration from environment variables."""
    containers_str = os.environ.get("CONTAINERS", "")
    containers = (
        [c.strip() for c in containers_str.split(",") if c.strip()]
        if containers_str
        else [
            "gen318-mqtt",
            "gen318-collector",
            "gen318-inference",
            "gen318-sync",
            "gen318-api",
        ]
    )

    return EdgeConfig(
        device_id=os.environ.get("DEVICE_ID", "edge-node-001"),
        cloud_api_url=os.environ.get("CLOUD_API_URL", ""),
        device_cert_path=os.environ.get("DEVICE_CERT_PATH", "/certs/device.pem"),
        device_key_path=os.environ.get("DEVICE_KEY_PATH", "/certs/device.key"),
        ca_cert_path=os.environ.get("CA_CERT_PATH", "/certs/gen318-ca.pem"),
        cert_pin_sha256=os.environ.get("CERT_PIN_SHA256", ""),
        sqlite_path=os.environ.get("SQLITE_PATH", "/data/gen318.db"),
        sqlite_encryption_key=os.environ.get("SQLITE_ENCRYPTION_KEY", ""),
        storage_target_days=int(os.environ.get("STORAGE_TARGET_DAYS", "30")),
        storage_expected_assets=int(os.environ.get("STORAGE_EXPECTED_ASSETS", "10")),
        storage_metrics_per_asset=int(
            os.environ.get("STORAGE_METRICS_PER_ASSET", "40")
        ),
        storage_bytes_per_measurement=int(
            os.environ.get("STORAGE_BYTES_PER_MEASUREMENT", "160")
        ),
        storage_reserve_bytes=int(
            os.environ.get("STORAGE_RESERVE_BYTES", str(2 * 1024**3))
        ),
        storage_warning_used_pct=float(
            os.environ.get("STORAGE_WARNING_USED_PCT", "80")
        ),
        storage_critical_used_pct=float(
            os.environ.get("STORAGE_CRITICAL_USED_PCT", "90")
        ),
        storage_emergency_used_pct=float(
            os.environ.get("STORAGE_EMERGENCY_USED_PCT", "96")
        ),
        local_retention_days=int(os.environ.get("LOCAL_RETENTION_DAYS", "45")),
        adapter_config_path=os.environ.get(
            "ADAPTER_CONFIG_PATH", "/config/adapters.yaml"
        ),
        model_path=os.environ.get("MODEL_PATH", "/models"),
        model_trusted_public_keys_json=os.environ.get(
            "MODEL_TRUSTED_PUBLIC_KEYS_JSON", "{}"
        ),
        model_artifact_max_bytes=int(
            os.environ.get("MODEL_ARTIFACT_MAX_BYTES", str(50 * 1024 * 1024))
        ),
        sync_interval_seconds=int(os.environ.get("SYNC_INTERVAL_SECONDS", "60")),
        backfill_max_batch_mb=int(os.environ.get("BACKFILL_MAX_BATCH_MB", "5")),
        telemetry_batch_size=int(os.environ.get("TELEMETRY_BATCH_SIZE", "1000")),
        telemetry_max_batches_per_cycle=int(
            os.environ.get("TELEMETRY_MAX_BATCHES_PER_CYCLE", "10")
        ),
        telemetry_recent_window_hours=int(
            os.environ.get("TELEMETRY_RECENT_WINDOW_HOURS", "24")
        ),
        telemetry_retention_hours=int(
            os.environ.get("TELEMETRY_RETENTION_HOURS", "1080")
        ),
        poll_interval_seconds=int(os.environ.get("POLL_INTERVAL_SECONDS", "60")),
        register_probe_enabled=os.environ.get(
            "EDGE_REGISTER_PROBE_ENABLED", "false"
        ).lower()
        in {"1", "true", "yes", "on"},
        register_probe_host=os.environ.get("EDGE_REGISTER_PROBE_HOST", ""),
        register_probe_port=int(os.environ.get("EDGE_REGISTER_PROBE_PORT", "502")),
        register_probe_ranges=os.environ.get(
            "EDGE_REGISTER_PROBE_RANGES", "225:280-310,225:325-335"
        ),
        register_probe_interval_seconds=int(
            os.environ.get("EDGE_REGISTER_PROBE_INTERVAL_SECONDS", "3600")
        ),
        inference_interval_seconds=int(
            os.environ.get("INFERENCE_INTERVAL_SECONDS", "300")
        ),
        check_interval=int(os.environ.get("CHECK_INTERVAL", "60")),
        tamper_gpio_pin=int(os.environ.get("TAMPER_GPIO_PIN", "17")),
        monitored_containers=containers,
        mqtt_host=os.environ.get("MQTT_HOST", "mosquitto"),
        mqtt_port=int(os.environ.get("MQTT_PORT", "1883")),
    )
