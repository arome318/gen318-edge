"""EdgeDatabase unit tests — 10 tests covering all CRUD operations."""

from __future__ import annotations

import time


class TestMeasurements:
    def test_insert_and_get_latest(self, edge_db) -> None:
        now = int(time.time() * 1000)
        edge_db.insert_measurement(now, "asset-1", "battery.soc_pct", 75.0, 0)
        edge_db.insert_measurement(now, "asset-1", "battery.voltage_v", 51.2, 0)
        edge_db.insert_measurement(now - 1000, "asset-1", "battery.soc_pct", 74.0, 0)

        latest = edge_db.get_latest_readings()
        assert len(latest) == 2
        soc = next(r for r in latest if r["metric_name"] == "battery.soc_pct")
        assert soc["value"] == 75.0
        assert soc["timestamp"] == now

    def test_batch_insert(self, edge_db) -> None:
        now = int(time.time() * 1000)
        batch = [
            (now + i, "asset-1", f"metric_{i}", float(i), 0) for i in range(10_000)
        ]
        count = edge_db.insert_measurements(batch)
        assert count == 10_000

    def test_readings_history_time_filter(self, edge_db) -> None:
        now = int(time.time() * 1000)
        old = now - 7200_000  # 2 hours ago
        edge_db.insert_measurement(now, "a1", "m1", 10.0, 0)
        edge_db.insert_measurement(old, "a1", "m1", 5.0, 0)

        history = edge_db.get_readings_history("a1", "m1", hours=1)
        assert len(history) == 1
        assert history[0]["value"] == 10.0

    def test_insert_or_ignore_dedup(self, edge_db) -> None:
        now = int(time.time() * 1000)
        edge_db.insert_measurement(now, "a1", "m1", 10.0, 0)
        # Same PK, different value — should be ignored (not replaced)
        edge_db.insert_measurement(now, "a1", "m1", 99.0, 0)
        latest = edge_db.get_latest_readings()
        assert latest[0]["value"] == 10.0  # Original preserved


class TestSyncQueue:
    def test_priority_ordering(self, edge_db) -> None:
        edge_db.enqueue_sync(3, {"type": "backfill"})
        edge_db.enqueue_sync(0, {"type": "critical"})
        edge_db.enqueue_sync(2, {"type": "normal"})

        p0 = edge_db.get_queue_items(priority=0)
        p3 = edge_db.get_queue_items(priority=3)
        assert len(p0) == 1
        assert p0[0]["payload"]["type"] == "critical"
        assert len(p3) == 1
        assert p3[0]["payload"]["type"] == "backfill"

    def test_dead_letter_after_max_attempts(self, edge_db) -> None:
        item_id = edge_db.enqueue_sync(2, {"type": "test"})
        for _ in range(10):
            edge_db.increment_attempt(item_id)

        items = edge_db.get_queue_items(priority=2)
        assert len(items) == 0  # Excluded after 10 attempts

        dead = edge_db.get_dead_letter_count()
        assert dead == 1

    def test_queue_depth_by_priority(self, edge_db) -> None:
        edge_db.enqueue_sync(0, {"a": 1})
        edge_db.enqueue_sync(0, {"a": 2})
        edge_db.enqueue_sync(2, {"a": 3})

        depth = edge_db.get_queue_depth_by_priority()
        assert depth["critical"] == 2
        assert depth["normal"] == 1
        assert depth["backfill"] == 0


class TestEventsAndModels:
    def test_events_severity_filter(self, edge_db) -> None:
        now = int(time.time() * 1000)
        edge_db.insert_event("e1", "alert", "critical", {"msg": "fire"}, now)
        edge_db.insert_event("e2", "info", "info", {"msg": "ok"}, now)

        critical = edge_db.get_events(severity="critical")
        assert len(critical) == 1
        assert critical[0]["severity"] == "critical"

        all_events = edge_db.get_events()
        assert len(all_events) == 2

    def test_model_state_crud(self, edge_db) -> None:
        edge_db.update_model_state("battery_health", "1.0.0", "a" * 64, 95.2)
        models = edge_db.get_model_state()
        assert len(models) == 1
        assert models[0]["model_name"] == "battery_health"
        assert models[0]["accuracy"] == 95.2

    def test_config_get_set(self, edge_db) -> None:
        assert edge_db.get_config("missing") is None
        edge_db.set_config("firmware_version", "1.0.0")
        assert edge_db.get_config("firmware_version") == "1.0.0"


class TestUnsyncedDrainState:
    def test_unsynced_bounds_and_ordering(self, edge_db) -> None:
        now = int(time.time() * 1000)
        old = now - 48 * 3600_000
        edge_db.insert_measurement(old, "a1", "m1", 1.0, 0)
        edge_db.insert_measurement(now, "a1", "m1", 2.0, 0)

        recent = edge_db.get_unsynced_measurements(since_ms=now - 3600_000)
        assert [r["value"] for r in recent] == [2.0]

        backfill = edge_db.get_unsynced_measurements(before_ms=now - 3600_000)
        assert [r["value"] for r in backfill] == [1.0]

        both = edge_db.get_unsynced_measurements()
        assert [r["value"] for r in both] == [1.0, 2.0]  # oldest first

    def test_mark_synced_excludes_from_drain(self, edge_db) -> None:
        now = int(time.time() * 1000)
        edge_db.insert_measurement(now, "a1", "m1", 1.0, 0)
        assert edge_db.count_unsynced_measurements() == 1

        edge_db.mark_measurements_synced([now], ["a1"], ["m1"])
        assert edge_db.get_unsynced_measurements() == []
        assert edge_db.count_unsynced_measurements() == 0

    def test_mark_rejected_excludes_but_preserves(self, edge_db) -> None:
        now = int(time.time() * 1000)
        edge_db.insert_measurement(now, "a1", "m1", 1.0, 0)
        edge_db.mark_measurements_rejected([now], ["a1"], ["m1"])

        assert edge_db.get_unsynced_measurements() == []
        # Data preserved locally for diagnosis
        assert edge_db.get_latest_readings()[0]["value"] == 1.0

    def test_prune_removes_only_old_terminal_rows(self, edge_db) -> None:
        now = int(time.time() * 1000)
        old = now - 100 * 3600_000
        edge_db.insert_measurement(old, "a1", "m1", 1.0, 0)  # old but UNSYNCED — kept
        edge_db.insert_measurement(old, "a1", "m2", 2.0, 0)
        edge_db.mark_measurements_synced([old], ["a1"], ["m2"])  # old + synced — pruned
        edge_db.insert_measurement(old, "a1", "m3", 3.0, 0)
        edge_db.mark_measurements_rejected(
            [old], ["a1"], ["m3"]
        )  # old + rejected — pruned
        edge_db.insert_measurement(now, "a1", "m4", 4.0, 0)
        edge_db.mark_measurements_synced(
            [now], ["a1"], ["m4"]
        )  # recent + synced — kept

        pruned = edge_db.prune_synced_measurements(older_than_ms=now - 72 * 3600_000)
        assert pruned == 2

        remaining = {
            row[0]
            for row in edge_db._conn.execute(
                "SELECT metric_name FROM measurements"
            ).fetchall()
        }
        assert remaining == {"m1", "m4"}
