"""SQLCipher-encrypted local database for edge node.

Uses sqlcipher3 for AES-256-CBC encryption. Performance PRAGMAs tuned
for >= 6,000 INSERT/s on RPi 5. Dead-letter handling after 10 failed
sync attempts prevents infinite retry loops.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger()

# Maximum sync attempts before item is dead-lettered
MAX_SYNC_ATTEMPTS = 10


class EdgeDatabase:
    """Thread-safe SQLCipher database wrapper for edge-side storage."""

    def __init__(self, db_path: str, encryption_key: str) -> None:
        self._db_path = db_path
        self._encryption_key = encryption_key
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Open database connection with encryption and performance PRAGMAs."""
        try:
            # Try sqlcipher3 first (preferred — prebuilt aarch64 wheels)
            import sqlcipher3  # type: ignore[import-untyped]

            self._conn = sqlcipher3.connect(self._db_path, check_same_thread=False)
        except ImportError:
            # Fall back to standard sqlite3 for development/testing
            logger.warning(
                "sqlcipher3_not_available",
                msg="Using unencrypted sqlite3 — not suitable for production",
            )
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)

        cursor = self._conn.cursor()

        # Encryption key (no-op on standard sqlite3)
        if self._encryption_key:
            try:
                # Use parameterized form to prevent injection.
                # SQLCipher PRAGMA key accepts a hex-encoded string.
                cursor.execute("PRAGMA key = ?", (self._encryption_key,))
                cursor.execute("PRAGMA cipher_page_size = 4096")
            except sqlite3.OperationalError:
                logger.warning(
                    "sqlcipher_pragma_key_unsupported",
                    msg="PRAGMA key not supported — database is unencrypted",
                )

        # Performance PRAGMAs — tuned for >= 6,000 inserts/s
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.execute("PRAGMA temp_store = MEMORY")
        cursor.execute("PRAGMA cache_size = -8000")  # 8 MB cache
        cursor.execute("PRAGMA mmap_size = 67108864")  # 64 MB mmap
        self._conn.commit()

        logger.info("edge_database_connected", db_path=self._db_path)

    def init_schema(self) -> None:
        """Create all tables if they don't exist."""
        if self._conn is None:
            raise RuntimeError("Database not connected")

        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                timestamp   INTEGER NOT NULL,
                asset_id    TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value       REAL NOT NULL,
                quality     INTEGER DEFAULT 0,
                synced      INTEGER DEFAULT 0,
                PRIMARY KEY (timestamp, asset_id, metric_name)
            ) WITHOUT ROWID
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_unsynced
                ON measurements(synced, timestamp)
                WHERE synced = 0
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          TEXT PRIMARY KEY,
                type        TEXT NOT NULL,
                severity    TEXT NOT NULL,
                payload     TEXT NOT NULL,
                timestamp   INTEGER NOT NULL,
                synced      INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_unsynced
                ON events(synced, timestamp)
                WHERE synced = 0
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_state (
                model_name  TEXT PRIMARY KEY,
                version     TEXT NOT NULL,
                sha256      TEXT NOT NULL,
                loaded_at   INTEGER NOT NULL,
                accuracy    REAL
            )
        """)
        model_state_columns = {
            str(row[1])
            for row in cursor.execute("PRAGMA table_info(model_state)").fetchall()
        }
        if "contract_json" not in model_state_columns:
            cursor.execute("ALTER TABLE model_state ADD COLUMN contract_json TEXT")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_queue (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                priority    INTEGER NOT NULL,
                payload     BLOB NOT NULL,
                created_at  INTEGER NOT NULL,
                attempts    INTEGER DEFAULT 0,
                last_attempt_at INTEGER
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sync_queue_priority
                ON sync_queue(priority, created_at)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  INTEGER NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS register_candidates (
                timestamp   INTEGER NOT NULL,
                unit_id     INTEGER NOT NULL,
                address     INTEGER NOT NULL,
                decoding    TEXT NOT NULL,
                value       REAL NOT NULL,
                PRIMARY KEY (timestamp, unit_id, address, decoding)
            ) WITHOUT ROWID
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_register_candidates_series
                ON register_candidates(unit_id, address, decoding, timestamp)
        """)

        # Vendor-map validation records (vendor integrations spec section 10).
        # Deliberately a SEPARATE table from register_candidates: a candidate
        # row is one numeric decoding of one address, and cannot hold an
        # identity block, wiring notes, a failed read, or the operator blanks
        # that turn a sweep into evidence. The record itself is JSON because
        # its shape is the spec's, not the database's, and a technician fills
        # half of it in by hand afterwards.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vendor_map_validations (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    INTEGER NOT NULL,
                map_name     TEXT NOT NULL,
                verification TEXT NOT NULL,
                record       TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_vendor_map_validations_map
                ON vendor_map_validations(map_name, timestamp)
        """)

        self._conn.commit()
        logger.info("edge_database_schema_initialized")

    # ── Measurements ─────────────────────────────────────────────

    def insert_measurement(
        self,
        timestamp: int,
        asset_id: str,
        metric_name: str,
        value: float,
        quality: int = 0,
    ) -> None:
        """Insert a single measurement."""
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT OR IGNORE INTO measurements "
                "(timestamp, asset_id, metric_name, value, quality) "
                "VALUES (?, ?, ?, ?, ?)",
                (timestamp, asset_id, metric_name, value, quality),
            )
            self._conn.commit()

    def insert_measurements(self, batch: list[tuple[int, str, str, float, int]]) -> int:
        """Batch-insert measurements for high throughput (>= 6,000 rows/s)."""
        with self._lock:
            assert self._conn is not None
            self._conn.executemany(
                "INSERT OR IGNORE INTO measurements "
                "(timestamp, asset_id, metric_name, value, quality) "
                "VALUES (?, ?, ?, ?, ?)",
                batch,
            )
            self._conn.commit()
        return len(batch)

    def get_latest_readings(self) -> list[dict[str, Any]]:
        """Get the latest reading for each (asset_id, metric_name) pair."""
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute("""
            SELECT m.asset_id, m.metric_name, m.value, m.quality, m.timestamp
            FROM measurements m
            INNER JOIN (
                SELECT asset_id, metric_name, MAX(timestamp) AS max_ts
                FROM measurements
                GROUP BY asset_id, metric_name
            ) latest ON m.asset_id = latest.asset_id
                    AND m.metric_name = latest.metric_name
                    AND m.timestamp = latest.max_ts
            ORDER BY m.asset_id, m.metric_name
        """)
            return [
                {
                    "asset_id": row[0],
                    "metric_name": row[1],
                    "value": row[2],
                    "quality": row[3],
                    "timestamp": row[4],
                }
                for row in cursor.fetchall()
            ]

    def get_series(self, asset_id: str, metric: str, hours: int = 24) -> tuple[Any, Any]:
        """Good-quality readings as two NumPy arrays (seconds, values), oldest first."""
        import numpy as np

        with self._lock:
            assert self._conn is not None
            cutoff = int((time.time() - hours * 3600) * 1000)
            rows = self._conn.execute(
                "SELECT timestamp, value FROM measurements "
                "WHERE asset_id = ? AND metric_name = ? AND timestamp >= ? AND quality <= 2 "
                "ORDER BY timestamp ASC",
                (asset_id, metric, cutoff),
            ).fetchall()
        if not rows:
            return np.array([]), np.array([])
        arr = np.asarray(rows, dtype=np.float64)
        keep = np.isfinite(arr[:, 1])
        return arr[keep, 0] / 1000.0, arr[keep, 1]

    def series_between(self, start_ms: int, end_ms: int, metrics: tuple[str, ...]
                       ) -> dict[tuple[str, str], tuple[Any, Any]]:
        """Good-quality readings of the given metrics in [start, end), per (asset, metric).

        For reports over a fixed period (a month's operating record), unlike
        ``get_series`` which looks back from now. Arrays are (seconds, values).
        """
        import numpy as np

        if not metrics:
            return {}
        marks = ",".join("?" * len(metrics))
        with self._lock:
            assert self._conn is not None
            rows = self._conn.execute(
                f"SELECT asset_id, metric_name, timestamp, value FROM measurements "
                f"WHERE timestamp >= ? AND timestamp < ? AND quality <= 2 AND metric_name IN ({marks}) "
                f"ORDER BY timestamp ASC",
                (start_ms, end_ms, *metrics),
            ).fetchall()
        grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for asset, metric, ts, value in rows:
            if value is not None and np.isfinite(value):
                grouped.setdefault((asset, metric), []).append((ts / 1000.0, float(value)))
        return {k: (np.array([t for t, _ in v]), np.array([x for _, x in v])) for k, v in grouped.items()}

    def get_readings_history(
        self, asset_id: str, metric: str, hours: int = 24
    ) -> list[dict[str, Any]]:
        """Get historical readings for a specific asset and metric."""
        with self._lock:
            assert self._conn is not None
            cutoff = int((time.time() - hours * 3600) * 1000)
            cursor = self._conn.execute(
                "SELECT timestamp, value, quality FROM measurements "
                "WHERE asset_id = ? AND metric_name = ? AND timestamp >= ? "
                "ORDER BY timestamp ASC",
                (asset_id, metric, cutoff),
            )
            return [
                {"timestamp": row[0], "value": row[1], "quality": row[2]}
                for row in cursor.fetchall()
            ]

    def get_unsynced_measurements(
        self,
        limit: int = 1000,
        since_ms: int | None = None,
        before_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get measurements pending sync, oldest first.

        ``since_ms``/``before_ms`` bound the timestamp range so the sync
        engine can drain recent telemetry (P2) separately from historical
        backfill (P3).
        """
        query = (
            "SELECT timestamp, asset_id, metric_name, value, quality "
            "FROM measurements WHERE synced = 0"
        )
        params: list[Any] = []
        if since_ms is not None:
            query += " AND timestamp >= ?"
            params.append(since_ms)
        if before_ms is not None:
            query += " AND timestamp < ?"
            params.append(before_ms)
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(query, params)
            return [
                {
                    "timestamp": row[0],
                    "asset_id": row[1],
                    "metric_name": row[2],
                    "value": row[3],
                    "quality": row[4],
                }
                for row in cursor.fetchall()
            ]

    def mark_measurements_synced(
        self, timestamps: list[int], asset_ids: list[str], metric_names: list[str]
    ) -> None:
        """Mark measurements as synced."""
        self._set_measurements_sync_state(1, timestamps, asset_ids, metric_names)

    def mark_measurements_rejected(
        self, timestamps: list[int], asset_ids: list[str], metric_names: list[str]
    ) -> None:
        """Mark measurements the cloud rejected as invalid (synced = 2).

        Rejected rows are excluded from future drains so a poison row cannot
        retry forever, but the data is preserved locally for diagnosis.
        """
        self._set_measurements_sync_state(2, timestamps, asset_ids, metric_names)

    def _set_measurements_sync_state(
        self,
        state: int,
        timestamps: list[int],
        asset_ids: list[str],
        metric_names: list[str],
    ) -> None:
        with self._lock:
            assert self._conn is not None
            self._conn.executemany(
                f"UPDATE measurements SET synced = {state} "
                "WHERE timestamp = ? AND asset_id = ? AND metric_name = ?",
                list(zip(timestamps, asset_ids, metric_names)),
            )
            self._conn.commit()

    def count_unsynced_measurements(self) -> int:
        """Count measurements still pending sync."""
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM measurements WHERE synced = 0"
            )
            return int(cursor.fetchone()[0])

    def count_rejected_measurements(self) -> int:
        """Count measurements the cloud rejected as invalid (synced = 2)."""
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM measurements WHERE synced = 2"
            )
            return int(cursor.fetchone()[0])

    def prune_synced_measurements(self, older_than_ms: int) -> int:
        """Delete synced/rejected measurements older than the retention cutoff.

        Rows still pending sync (synced = 0) are NEVER pruned — offline
        backlog is preserved until the cloud confirms it. Without this sweep
        the WITHOUT ROWID table grows unboundedly on the Pi's SD card.
        """
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "DELETE FROM measurements WHERE synced != 0 AND timestamp < ?",
                (older_than_ms,),
            )
            self._conn.commit()
            return cursor.rowcount

    def prune_local_history(self, older_than_ms: int) -> dict[str, int]:
        """Standalone boxes only: delete everything older than the cutoff.

        With no server configured nothing is ever marked synced, so
        ``prune_synced_measurements`` never frees a row and the disk fills.
        A standalone box has nowhere to send old rows; it keeps what its own
        models and pages need and lets the rest go. Never call this on a box
        that syncs: there the unsynced backlog is the only copy.
        """
        with self._lock:
            assert self._conn is not None
            counts = {}
            for table, column in (("measurements", "timestamp"), ("events", "timestamp"),
                                  ("sync_queue", "created_at")):
                cursor = self._conn.execute(
                    f"DELETE FROM {table} WHERE {column} < ?", (older_than_ms,)
                )
                counts[table] = cursor.rowcount
            self._conn.commit()
            return counts

    # ── Events ───────────────────────────────────────────────────

    def insert_event(
        self,
        event_id: str,
        event_type: str,
        severity: str,
        payload: dict[str, Any],
        timestamp: int,
    ) -> None:
        """Insert a local event."""
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT OR REPLACE INTO events "
                "(id, type, severity, payload, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (event_id, event_type, severity, json.dumps(payload), timestamp),
            )
            self._conn.commit()

    def get_events(
        self, severity: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get recent events with optional severity filter."""
        with self._lock:
            assert self._conn is not None
            if severity:
                cursor = self._conn.execute(
                    "SELECT id, type, severity, payload, timestamp, synced "
                    "FROM events WHERE severity = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (severity, limit),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT id, type, severity, payload, timestamp, synced "
                    "FROM events ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            return [
                {
                    "id": row[0],
                    "type": row[1],
                    "severity": row[2],
                    "payload": json.loads(row[3]),
                    "timestamp": row[4],
                    "synced": bool(row[5]),
                }
                for row in cursor.fetchall()
            ]

    # ── Register probe candidates (mapping evidence, not telemetry) ──

    def insert_register_candidates(
        self, batch: list[tuple[int, int, int, str, float]]
    ) -> int:
        """Batch-insert register decoding candidates.

        Rows are (timestamp_ms, unit_id, address, decoding, value).
        INSERT OR IGNORE on the PK keeps re-probes idempotent.
        """
        with self._lock:
            assert self._conn is not None
            self._conn.executemany(
                "INSERT OR IGNORE INTO register_candidates "
                "(timestamp, unit_id, address, decoding, value) "
                "VALUES (?, ?, ?, ?, ?)",
                batch,
            )
            self._conn.commit()
        return len(batch)

    def get_register_candidate_series(
        self, unit_id: int, address: int, decoding: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Newest-first series for one (unit, address, decoding) candidate."""
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "SELECT timestamp, value FROM register_candidates "
                "WHERE unit_id = ? AND address = ? AND decoding = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (unit_id, address, decoding, limit),
            )
            return [{"timestamp": row[0], "value": row[1]} for row in cursor.fetchall()]

    def prune_register_candidates(self, older_than_ms: int) -> int:
        """Delete candidate rows older than the retention cutoff.

        Candidates are local mapping evidence, not synced telemetry —
        unconditional age-based pruning is safe here.
        """
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "DELETE FROM register_candidates WHERE timestamp < ?",
                (older_than_ms,),
            )
            self._conn.commit()
            return cursor.rowcount

    # ── Vendor map validation records (spec section 10) ──────────

    def insert_vendor_map_validation(
        self, timestamp: int, map_name: str, verification: str, record: dict[str, Any]
    ) -> int:
        """Store one validation record. Returns its row id.

        Append-only by convention: a later sweep is a NEW record, never an
        edit of an old one. A record describes one unit at one moment, and
        overwriting it would destroy the comparison a bench gate depends on.
        """
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "INSERT INTO vendor_map_validations "
                "(timestamp, map_name, verification, record) VALUES (?, ?, ?, ?)",
                (timestamp, map_name, verification, json.dumps(record)),
            )
            self._conn.commit()
            return cursor.lastrowid or 0

    def get_vendor_map_validations(
        self, map_name: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Newest-first validation records, optionally for one map."""
        with self._lock:
            assert self._conn is not None
            if map_name is None:
                cursor = self._conn.execute(
                    "SELECT id, timestamp, map_name, verification, record "
                    "FROM vendor_map_validations ORDER BY timestamp DESC, id DESC "
                    "LIMIT ?",
                    (limit,),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT id, timestamp, map_name, verification, record "
                    "FROM vendor_map_validations WHERE map_name = ? "
                    "ORDER BY timestamp DESC, id DESC LIMIT ?",
                    (map_name, limit),
                )
            return [
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "map_name": row[2],
                    "verification": row[3],
                    "record": json.loads(row[4]),
                }
                for row in cursor.fetchall()
            ]

    # ── Sync Queue ───────────────────────────────────────────────

    def enqueue_sync(self, priority: int, payload: dict[str, Any]) -> int:
        """Add an item to the sync queue with gzip compression."""
        compressed = gzip.compress(json.dumps(payload).encode(), compresslevel=6)
        now = int(time.time() * 1000)
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "INSERT INTO sync_queue (priority, payload, created_at) "
                "VALUES (?, ?, ?)",
                (priority, compressed, now),
            )
            self._conn.commit()
            return cursor.lastrowid or 0

    def get_queue_items(self, priority: int, limit: int = 1000) -> list[dict[str, Any]]:
        """Get sync queue items for a priority tier, excluding dead-lettered."""
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "SELECT id, priority, payload, created_at, attempts "
                "FROM sync_queue "
                "WHERE priority = ? AND attempts < ? "
                "ORDER BY created_at ASC LIMIT ?",
                (priority, MAX_SYNC_ATTEMPTS, limit),
            )
            return [
                {
                    "id": row[0],
                    "priority": row[1],
                    "payload": json.loads(gzip.decompress(row[2]).decode()),
                    "created_at": row[3],
                    "attempts": row[4],
                }
                for row in cursor.fetchall()
            ]

    def mark_synced(self, ids: list[int]) -> None:
        """Remove successfully synced items from the queue."""
        with self._lock:
            assert self._conn is not None
            placeholders = ",".join("?" for _ in ids)
            self._conn.execute(
                f"DELETE FROM sync_queue WHERE id IN ({placeholders})",
                ids,
            )
            self._conn.commit()

    def increment_attempt(self, item_id: int) -> None:
        """Increment the attempt counter for a failed sync item."""
        now = int(time.time() * 1000)
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "UPDATE sync_queue SET attempts = attempts + 1, "
                "last_attempt_at = ? WHERE id = ?",
                (now, item_id),
            )
            self._conn.commit()

    def get_queue_depth_by_priority(self) -> dict[str, int]:
        """Get queue depth per priority tier."""
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "SELECT priority, COUNT(*) FROM sync_queue "
                "WHERE attempts < ? GROUP BY priority",
                (MAX_SYNC_ATTEMPTS,),
            )
            priority_names = {0: "critical", 1: "high", 2: "normal", 3: "backfill"}
            result = {name: 0 for name in priority_names.values()}
            for row in cursor.fetchall():
                name = priority_names.get(row[0], f"p{row[0]}")
                result[name] = row[1]
            return result

    def get_dead_letter_count(self) -> int:
        """Count items that have exceeded maximum sync attempts."""
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE attempts >= ?",
                (MAX_SYNC_ATTEMPTS,),
            )
            return cursor.fetchone()[0]

    # ── Model State ──────────────────────────────────────────────

    def get_model_state(self) -> list[dict[str, Any]]:
        """Get loaded model versions and accuracy."""
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "SELECT model_name, version, sha256, loaded_at, accuracy, contract_json "
                "FROM model_state ORDER BY model_name"
            )
            return [
                {
                    "model_name": row[0],
                    "version": row[1],
                    "sha256": row[2],
                    "loaded_at": row[3],
                    "accuracy": row[4],
                    "feature_contract": json.loads(row[5]) if row[5] else None,
                }
                for row in cursor.fetchall()
            ]

    def update_model_state(
        self,
        model_name: str,
        version: str,
        sha256: str,
        accuracy: float | None = None,
        feature_contract: dict[str, Any] | None = None,
    ) -> None:
        """Update or insert model state."""
        now = int(time.time() * 1000)
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT OR REPLACE INTO model_state "
                "(model_name, version, sha256, loaded_at, accuracy, contract_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    model_name,
                    version,
                    sha256,
                    now,
                    accuracy,
                    json.dumps(feature_contract, sort_keys=True)
                    if feature_contract
                    else None,
                ),
            )
            self._conn.commit()

    # ── Config ───────────────────────────────────────────────────

    def get_latest_reading(
        self, metric_name: str, asset_id: str | None = None
    ) -> dict[str, Any] | None:
        query = (
            "SELECT asset_id, metric_name, value, quality, timestamp FROM measurements "
            "WHERE metric_name = ?"
        )
        params: list[Any] = [metric_name]
        if asset_id:
            query += " AND asset_id = ?"
            params.append(asset_id)
        query += " ORDER BY timestamp DESC LIMIT 1"
        with self._lock:
            assert self._conn is not None
            row = self._conn.execute(query, params).fetchone()
        if not row:
            return None
        return {
            "asset_id": row[0],
            "metric_name": row[1],
            "value": row[2],
            "quality": row[3],
            "timestamp": row[4],
        }

    def get_config(self, key: str) -> str | None:
        """Get a configuration value."""
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "SELECT value FROM config WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def set_config(self, key: str, value: str) -> None:
        """Set a configuration value."""
        now = int(time.time() * 1000)
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT OR REPLACE INTO config (key, value, updated_at) "
                "VALUES (?, ?, ?)",
                (key, value, now),
            )
            self._conn.commit()

    # ── Lifecycle ────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("edge_database_closed")
