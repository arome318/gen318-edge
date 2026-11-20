"""Edge collector service — orchestrates hardware adapters (Doc 14 + Doc 15).

Runs as the gen318-collector container. Instantiates hardware adapters
from YAML configuration, manages their lifecycle via ManagedAdapter
wrappers, and supervises concurrent poll/subscribe tasks with automatic
restart on failure.

Readings flow: Adapter → ManagedAdapter → SQLite + MQTT
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import yaml

from collector.adapters.base import AdapterStatus, Reading
from collector.adapters.registry import ADAPTER_REGISTRY
from collector.pq_detector import PQDetector
from collector.register_probe import RegisterProbe
from common.config import EdgeConfig, load_config
from common.database import EdgeDatabase
from common.mqtt_client import (
    TOPIC_CONTROL_ADAPTER_RECONNECT,
    TOPIC_TELEMETRY,
    MQTTClient,
)
from common.storage import StoragePolicy, StoragePressure, assess_storage

logger = structlog.get_logger()


# ── ManagedAdapter ──────────────────────────────────────────────


@dataclass
class ManagedAdapter:
    """Wraps a HardwareAdapter with operational metadata and status tracking.

    Tracks error counts, reading counts, and last reading timestamp.
    The collector uses these to report status and make supervision decisions.
    """

    name: str
    adapter_type: str
    config: dict[str, Any]
    status: AdapterStatus = AdapterStatus.DISCONNECTED
    error_count: int = 0
    last_reading_at: float = 0.0
    reading_count: int = 0
    _instance: Any = field(default=None, repr=False)

    @property
    def instance(self) -> Any:
        return self._instance

    @instance.setter
    def instance(self, value: Any) -> None:
        self._instance = value

    def record_readings(self, count: int) -> None:
        """Record successful readings."""
        self.reading_count += count
        self.last_reading_at = time.time()
        self.error_count = 0
        self.status = AdapterStatus.CONNECTED

    def record_error(self) -> None:
        """Record an adapter error."""
        self.error_count += 1
        if self.error_count > 10:
            self.status = AdapterStatus.ERROR

    def to_status_dict(self) -> dict[str, Any]:
        """Serialize status for config table / API consumption."""
        return {
            "name": self.name,
            "type": self.adapter_type,
            "status": self.status.value,
            "error_count": self.error_count,
            "last_reading_at": self.last_reading_at,
            "reading_count": self.reading_count,
        }


# ── AdapterSupervisor ──────────────────────────────────────────


class AdapterSupervisor:
    """Supervises adapter tasks with automatic restart on failure.

    Unlike asyncio.gather (no restart) or TaskGroup (cancels siblings),
    this supervisor monitors each task independently and restarts failed
    ones with exponential backoff up to a maximum restart count.
    """

    def __init__(self, max_restarts: int = 10) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._factories: dict[str, Any] = {}  # name → coroutine factory
        self._restart_counts: dict[str, int] = {}
        self._max_restarts = max_restarts
        self._running = True

    def add(self, name: str, coro_factory: Any) -> None:
        """Register a task factory. The factory is a zero-arg async callable."""
        self._factories[name] = coro_factory
        self._restart_counts[name] = 0

    async def run(self) -> None:
        """Start all tasks and supervise them."""
        # Launch initial tasks
        for name, factory in self._factories.items():
            self._tasks[name] = asyncio.create_task(factory(), name=f"adapter-{name}")

        # Supervision loop
        while self._running and self._tasks:
            done, _pending = await asyncio.wait(
                self._tasks.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                # Find which adapter this task belongs to
                task_name = None
                for name, t in self._tasks.items():
                    if t is task:
                        task_name = name
                        break

                if task_name is None:
                    continue

                exc = task.exception() if not task.cancelled() else None

                if exc is not None:
                    self._restart_counts[task_name] += 1
                    count = self._restart_counts[task_name]

                    if count <= self._max_restarts:
                        backoff = min(2**count, 60)
                        logger.warning(
                            "adapter_task_failed",
                            adapter=task_name,
                            error=str(exc),
                            restart_count=count,
                            backoff_s=backoff,
                        )
                        # Non-blocking restart: wrap in a delayed-start task
                        # so we don't block supervision of other adapters
                        if self._running and task_name in self._factories:
                            factory = self._factories[task_name]
                            self._tasks[task_name] = asyncio.create_task(
                                self._delayed_start(factory, backoff),
                                name=f"adapter-{task_name}",
                            )
                    else:
                        logger.error(
                            "adapter_task_max_restarts",
                            adapter=task_name,
                            max_restarts=self._max_restarts,
                        )
                        del self._tasks[task_name]
                else:
                    # Task completed normally (shouldn't happen for adapters)
                    logger.info("adapter_task_completed", adapter=task_name)
                    del self._tasks[task_name]

    @staticmethod
    async def _delayed_start(factory: Any, delay: float) -> None:
        """Wait before restarting a failed adapter task."""
        await asyncio.sleep(delay)
        await factory()

    async def stop(self) -> None:
        """Cancel all supervised tasks."""
        self._running = False
        tasks_to_cancel = list(self._tasks.items())
        for name, task in tasks_to_cancel:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
        logger.info("supervisor_stopped")


# ── CollectorService ───────────────────────────────────────────


class CollectorService:
    """Orchestrates all configured hardware adapters.

    Instantiates adapters from YAML config, runs them as concurrent
    async tasks (poll-based or subscribe-based), writes readings to
    SQLite, and publishes to local MQTT for the inference container.
    """

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self.db = EdgeDatabase(config.sqlite_path, config.sqlite_encryption_key)
        self.mqtt = MQTTClient(
            host=config.mqtt_host,
            port=config.mqtt_port,
            client_id="gen318-collector",
        )
        self._managed: list[ManagedAdapter] = []
        self._pq_detector = PQDetector()
        self._supervisor = AdapterSupervisor(max_restarts=10)
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_storage_check_at = 0.0
        self._last_storage_pressure: StoragePressure | None = None

    def _storage_policy(self) -> StoragePolicy:
        return StoragePolicy(
            target_days=self.config.storage_target_days,
            expected_assets=self.config.storage_expected_assets,
            metrics_per_asset=self.config.storage_metrics_per_asset,
            bytes_per_measurement=self.config.storage_bytes_per_measurement,
            poll_interval_seconds=self.config.poll_interval_seconds,
            reserve_bytes=self.config.storage_reserve_bytes,
            warning_used_pct=self.config.storage_warning_used_pct,
            critical_used_pct=self.config.storage_critical_used_pct,
            emergency_used_pct=self.config.storage_emergency_used_pct,
        )

    def _maintain_storage(self, *, force: bool = False) -> None:
        """Publish runway and reclaim only cloud-confirmed telemetry.

        This routine is best-effort by design. A filesystem-stat failure must
        never stop hardware collection, and no pressure state is authorized to
        delete an unsynced row.
        """
        now = time.time()
        if not force and now - self._last_storage_check_at < 60:
            return
        self._last_storage_check_at = now

        try:
            status = assess_storage(self.config.sqlite_path, self._storage_policy())
            self.db.set_config("storage.status", json.dumps(status.to_dict()))

            if status.pressure in {
                StoragePressure.CRITICAL,
                StoragePressure.EMERGENCY,
            }:
                # Preserve the most recent hour for local dashboards while
                # aggressively reclaiming only rows already acknowledged by
                # (or permanently rejected by) the cloud.
                cutoff_ms = int((now - 3600) * 1000)
                pruned = self.db.prune_synced_measurements(cutoff_ms)
                if pruned:
                    logger.warning(
                        "storage_pressure_pruned_synced",
                        pressure=status.pressure.value,
                        pruned=pruned,
                        runway_days=status.estimated_runway_days,
                    )

            if status.pressure != self._last_storage_pressure:
                logger.warning(
                    "storage_pressure_changed",
                    previous=(
                        self._last_storage_pressure.value
                        if self._last_storage_pressure
                        else None
                    ),
                    current=status.pressure.value,
                    runway_days=status.estimated_runway_days,
                    free_bytes=status.free_bytes,
                )
                self._last_storage_pressure = status.pressure
        except Exception:
            logger.exception("storage_assessment_failed")

    def _load_adapter_config(self) -> list[dict[str, Any]]:
        """Load adapter definitions from YAML config file."""
        config_path = Path(self.config.adapter_config_path)
        if not config_path.exists():
            logger.warning(
                "adapter_config_not_found",
                path=str(config_path),
                msg="Using empty adapter list — no data collection",
            )
            return []

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        adapters = raw.get("adapters", [])
        logger.info("adapters_loaded", count=len(adapters))
        return adapters

    async def _instantiate_adapters(self) -> None:
        """Create adapter instances from YAML configuration."""
        adapter_configs = self._load_adapter_config()

        for adapter_conf in adapter_configs:
            if not adapter_conf.get("enabled", True):
                logger.info(
                    "adapter_disabled",
                    name=adapter_conf.get("name", "unknown"),
                )
                continue

            name = adapter_conf["name"]
            adapter_type = adapter_conf["type"]
            config = adapter_conf.get("config", {})

            if adapter_type not in ADAPTER_REGISTRY:
                logger.error(
                    "unknown_adapter_type",
                    name=name,
                    type=adapter_type,
                    available=list(ADAPTER_REGISTRY.keys()),
                )
                continue

            managed = ManagedAdapter(
                name=name,
                adapter_type=adapter_type,
                config=config,
            )

            try:
                cls = ADAPTER_REGISTRY[adapter_type]
                instance = cls()
                ok = await instance.initialize(config)

                if ok:
                    managed.instance = instance
                    managed.status = AdapterStatus.CONNECTED
                    logger.info(
                        "adapter_initialized",
                        name=name,
                        type=adapter_type,
                    )
                    self._queue_vendor_map_evidence(name, instance)
                else:
                    managed.status = AdapterStatus.ERROR
                    logger.warning(
                        "adapter_init_failed",
                        name=name,
                        type=adapter_type,
                    )
            except Exception:
                managed.status = AdapterStatus.ERROR
                logger.exception(
                    "adapter_init_exception",
                    name=name,
                    type=adapter_type,
                )

            self._managed.append(managed)
            self._update_adapter_status(managed)

    def _queue_vendor_map_evidence(self, name: str, instance: Any) -> None:
        """Ship a vendor-map adapter's provenance and identity as tier-3 evidence.

        A validation record (spec §10) is about a UNIT, not a brand: register
        maps drift by firmware, so the record has to name the map revision, its
        verification tier and the serial/model the device reported. This is the
        only place that leaves the edge, and it leaves as priority 3 — behind
        alerts and telemetry, drained only on a healthy link.

        Offline-first: this enqueues and returns. It never awaits the cloud and
        never fails an adapter that has already initialized.
        """
        summary_fn = getattr(instance, "vendor_map_summary", None)
        if not callable(summary_fn):
            return  # legacy adapter, or one with no map
        try:
            summary = summary_fn()
            if not summary:
                return
            self.db.enqueue_sync(
                priority=3,
                payload={
                    "type": "edge.vendor_map_identity",
                    "title": "Vendor map identity",
                    "adapter": name,
                    "device_id": self.config.device_id,
                    "observed_at": datetime.now(UTC).isoformat(),
                    **summary,
                },
            )
            logger.info(
                "vendor_map_evidence_queued",
                adapter=name,
                vendor_map=summary.get("name"),
                verification=summary.get("verification"),
            )
        except Exception:
            logger.exception("vendor_map_evidence_enqueue_failed", adapter=name)

    def _update_adapter_status(self, managed: ManagedAdapter) -> None:
        """Write adapter status to the config table for API consumption."""
        key = f"adapter.{managed.name}.status"
        value = json.dumps(managed.to_status_dict())
        self.db.set_config(key, value)

    async def _on_readings(
        self, readings: list[Reading], managed: ManagedAdapter
    ) -> None:
        """Process readings from an adapter — write to SQLite and MQTT."""
        if not readings:
            return

        self._maintain_storage()

        batch: list[tuple[int, str, str, float, int]] = []

        for reading in readings:
            timestamp_ms = int(reading.timestamp.timestamp() * 1000)
            entry = (
                timestamp_ms,
                reading.asset_id or "unknown",
                reading.metric_name,
                reading.value,
                reading.quality,
            )
            batch.append(entry)

            # Publish to local MQTT for inference container
            self.mqtt.publish(
                TOPIC_TELEMETRY,
                {
                    "timestamp": timestamp_ms,
                    "asset_id": reading.asset_id,
                    "metric_name": reading.metric_name,
                    "value": reading.value,
                    "quality": reading.quality,
                    "device_id": self.config.device_id,
                },
            )

        if batch:
            inserted = self.db.insert_measurements(batch)
            managed.record_readings(len(batch))
            self._update_adapter_status(managed)
            logger.debug(
                "readings_stored",
                adapter=managed.name,
                count=inserted,
            )

        self._detect_power_quality(readings)

    def _detect_power_quality(self, readings: list[Reading]) -> None:
        """Feed readings to the PQ excursion detector and queue any events.

        Best-effort and isolated from the collection path on purpose (Doc 15
        invariant 1): a detector bug must never block or fail telemetry
        storage, so this always runs AFTER readings are already durable in
        SQLite and never raises.
        """
        try:
            events = self._pq_detector.process_batch(readings)
        except Exception:
            logger.exception("pq_detection_failed")
            return
        for event in events:
            # Fix round 1 / CRITICAL-1: the shared detector now always
            # builds a CLOSE event for every event_type, including
            # supply_interruption (needed so the cloud task can act on it —
            # see backend/app/tasks/pq_tasks.py). Edge behavior stays as
            # ruled: an interruption gets exactly ONE queue item, at OPEN,
            # priority 0 (the idempotency key it claims is INSERT ... ON
            # CONFLICT DO NOTHING, so a second message for the same
            # started_at would be silently dropped anyway) — its CLOSE
            # event is real and correctly built, but must never reach the
            # sync queue.
            if (
                event["event_type"] == "supply_interruption"
                and event["ended_at"] is not None
            ):
                continue
            try:
                priority = event.pop("priority", 1)
                self.db.enqueue_sync(priority=priority, payload=event)
            except Exception:
                logger.exception(
                    "pq_event_enqueue_failed", event_type=event.get("event_type")
                )

    async def _run_poll_adapter(self, managed: ManagedAdapter) -> None:
        """Poll loop for a single adapter."""
        adapter = managed.instance
        caps = adapter.capabilities()
        interval = max(caps.poll_interval_seconds, caps.min_poll_interval_seconds)

        while self._running:
            try:
                readings = await adapter.poll()
                if readings:
                    await self._on_readings(readings, managed)
            except Exception:
                managed.record_error()
                self._update_adapter_status(managed)
                logger.exception(
                    "adapter_poll_error",
                    adapter=managed.name,
                )

            await asyncio.sleep(interval)

    async def _run_subscribe_adapter(self, managed: ManagedAdapter) -> None:
        """Subscribe task for a single adapter."""
        adapter = managed.instance

        async def callback(readings: list[Reading]) -> None:
            await self._on_readings(readings, managed)

        try:
            await adapter.subscribe(callback)
        except Exception:
            managed.record_error()
            self._update_adapter_status(managed)
            raise  # Let supervisor handle restart

    def _on_reconnect_command(self, payload: dict[str, Any]) -> None:
        """Handle adapter reconnect command from MQTT.

        Called from paho-mqtt's background thread, so must use
        thread-safe scheduling to the asyncio event loop.
        """
        adapter_name = payload.get("adapter_name", "")
        logger.info("reconnect_command_received", adapter=adapter_name)

        for managed in self._managed:
            if managed.name == adapter_name and managed.instance:
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(
                        self._loop.create_task,
                        self._reconnect_adapter(managed),
                    )

    def _register_probe_task(self) -> None:
        """Register the opt-in register probe with the supervisor.

        Default OFF (EDGE_REGISTER_PROBE_ENABLED). The probe owns its own
        Modbus client and never touches measurements or MQTT — its failures
        cannot affect telemetry collection.
        """
        if not self.config.register_probe_enabled:
            return
        if not self.config.register_probe_host:
            logger.warning(
                "register_probe_host_missing",
                msg="EDGE_REGISTER_PROBE_ENABLED is true but "
                "EDGE_REGISTER_PROBE_HOST is empty — probe stays off",
            )
            return
        probe = RegisterProbe(
            db=self.db,
            host=self.config.register_probe_host,
            port=self.config.register_probe_port,
            ranges=self.config.register_probe_ranges,
            interval_seconds=self.config.register_probe_interval_seconds,
        )
        self._supervisor.add("register-probe", probe.run)
        logger.info(
            "register_probe_registered",
            host=self.config.register_probe_host,
            ranges=self.config.register_probe_ranges,
            interval_seconds=self.config.register_probe_interval_seconds,
        )

    async def _reconnect_adapter(self, managed: ManagedAdapter) -> None:
        """Reinitialize an adapter."""
        try:
            if managed.instance:
                await managed.instance.shutdown()

            cls = ADAPTER_REGISTRY[managed.adapter_type]
            instance = cls()
            ok = await instance.initialize(managed.config)

            if ok:
                managed.instance = instance
                managed.status = AdapterStatus.CONNECTED
                managed.error_count = 0
                logger.info("adapter_reconnected", name=managed.name)
            else:
                managed.status = AdapterStatus.ERROR
                logger.warning("adapter_reconnect_failed", name=managed.name)

        except Exception:
            managed.status = AdapterStatus.ERROR
            logger.exception("adapter_reconnect_error", name=managed.name)

        self._update_adapter_status(managed)

    def get_adapter_statuses(self) -> list[dict[str, Any]]:
        """Return status dicts for all managed adapters."""
        return [m.to_status_dict() for m in self._managed]

    async def run(self) -> None:
        """Main collector loop."""
        self._loop = asyncio.get_running_loop()
        self.db.connect()
        self.db.init_schema()
        self._maintain_storage(force=True)
        self.mqtt.connect()

        # Subscribe to reconnect commands
        self.mqtt.subscribe(
            TOPIC_CONTROL_ADAPTER_RECONNECT,
            self._on_reconnect_command,
        )

        # Instantiate all adapters
        await self._instantiate_adapters()
        self._running = True

        self._register_probe_task()

        # Register adapter tasks with supervisor
        for managed in self._managed:
            if managed.instance is None:
                continue

            caps = managed.instance.capabilities()
            if caps.can_subscribe:
                self._supervisor.add(
                    managed.name,
                    lambda m=managed: self._run_subscribe_adapter(m),
                )
            else:
                self._supervisor.add(
                    managed.name,
                    lambda m=managed: self._run_poll_adapter(m),
                )

        logger.info(
            "collector_started",
            device_id=self.config.device_id,
            adapter_count=len(self._managed),
            active_count=sum(1 for m in self._managed if m.instance is not None),
        )

        # Run supervisor (blocks until stopped)
        await self._supervisor.run()

    async def stop(self) -> None:
        """Gracefully stop the collector."""
        self._running = False

        # Stop supervisor (cancels all tasks)
        await self._supervisor.stop()

        # Shutdown all adapters
        for managed in self._managed:
            if managed.instance:
                try:
                    await managed.instance.shutdown()
                except Exception:
                    logger.exception(
                        "adapter_shutdown_error",
                        adapter=managed.name,
                    )

        self.mqtt.disconnect()
        self.db.close()
        logger.info("collector_stopped")


async def main() -> None:
    config = load_config()
    service = CollectorService(config)
    try:
        await service.run()
    except KeyboardInterrupt:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
