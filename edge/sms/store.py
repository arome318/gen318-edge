"""Local record of SMS alerts, deliveries, technician labels and preferences.

Phone numbers are never stored in clear: they are kept as salted SHA-256
hashes, so the record can be shared for research without exposing anyone.

The text of each SMS the box sends is kept, so the local page can show what
was said. The text of what people send in is not: only the kind of message
(a status request, a label) is recorded.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from typing import Any

_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS alerts (
    ref TEXT PRIMARY KEY, code TEXT NOT NULL, severity TEXT NOT NULL,
    asset_id TEXT NOT NULL, raised_at REAL NOT NULL, cleared_at REAL,
    values_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS deliveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ref TEXT, phone_hash TEXT NOT NULL,
    role TEXT NOT NULL, kind TEXT NOT NULL, sent_at REAL NOT NULL, ok INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ref TEXT NOT NULL, label INTEGER NOT NULL,
    note TEXT NOT NULL DEFAULT '', phone_hash TEXT NOT NULL, received_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS prefs (phone_hash TEXT PRIMARY KEY, lang TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS inbound (
    id INTEGER PRIMARY KEY AUTOINCREMENT, phone_hash TEXT NOT NULL,
    kind TEXT NOT NULL, received_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inbound_phone ON inbound(phone_hash, received_at);
CREATE INDEX IF NOT EXISTS idx_deliveries_phone ON deliveries(phone_hash, sent_at);
"""


def normalise_phone(phone: str) -> str:
    """Nigerian numbers to +234 form; others keep their digits and plus."""
    digits = "".join(ch for ch in phone if ch.isdigit() or ch == "+")
    if digits.startswith("+"):
        return digits
    if digits.startswith("234"):
        return "+" + digits
    if digits.startswith("0") and len(digits) == 11:
        return "+234" + digits[1:]
    return digits


class SmsStore:
    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            columns = {r[1] for r in self._conn.execute("PRAGMA table_info(deliveries)")}
            if "text" not in columns:
                try:
                    self._conn.execute("ALTER TABLE deliveries ADD COLUMN text TEXT NOT NULL DEFAULT ''")
                    self._conn.commit()
                except sqlite3.OperationalError:
                    pass  # opened read-only (local page) before the SMS service upgraded it
            row = self._conn.execute("SELECT value FROM meta WHERE key='salt'").fetchone()
            if row is None:
                self._salt = secrets.token_hex(16)
                self._conn.execute("INSERT INTO meta VALUES ('salt', ?)", (self._salt,))
                self._conn.commit()
            else:
                self._salt = row[0]

    def phone_hash(self, phone: str) -> str:
        return hashlib.sha256(f"{self._salt}:{normalise_phone(phone)}".encode()).hexdigest()

    def new_ref(self) -> str:
        while True:
            ref = "".join(secrets.choice(_ALPHABET) for _ in range(4))
            with self._lock:
                hit = self._conn.execute("SELECT 1 FROM alerts WHERE ref=?", (ref,)).fetchone()
            if not hit:
                return ref

    def record_alert(self, ref: str, code: str, severity: str, asset_id: str,
                     values: dict[str, Any], now: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO alerts (ref, code, severity, asset_id, raised_at, values_json) "
                "VALUES (?,?,?,?,?,?)",
                (ref, code, severity, asset_id, now, json.dumps(values)),
            )
            self._conn.commit()

    def clear_alert(self, code: str, asset_id: str, now: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE alerts SET cleared_at=? WHERE code=? AND asset_id=? AND cleared_at IS NULL",
                (now, code, asset_id),
            )
            self._conn.commit()

    def record_delivery(self, ref: str | None, phone: str, role: str, kind: str,
                        ok: bool, now: float, text: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO deliveries (ref, phone_hash, role, kind, sent_at, ok, text) "
                "VALUES (?,?,?,?,?,?,?)",
                (ref, self.phone_hash(phone), role, kind, now, int(ok), text),
            )
            self._conn.commit()

    def recent_messages(self, limit: int = 20) -> list[dict[str, Any]]:
        """SMS sent and received, newest first, for the local page (no phone numbers).

        Sent messages carry their text. Received ones carry only their kind.
        """
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT at, direction, role, kind, text, ok FROM ("
                    " SELECT sent_at AS at, 'out' AS direction, role, kind, text, ok FROM deliveries"
                    " UNION ALL"
                    " SELECT received_at, 'in', '', kind, '', 1 FROM inbound"
                    ") ORDER BY at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        keys = ("at", "direction", "role", "kind", "text", "ok")
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def latest_alert_ref_for(self, phone: str, since: float) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT ref FROM deliveries WHERE phone_hash=? AND kind='alert' AND sent_at>=? "
                "ORDER BY sent_at DESC, id DESC LIMIT 1",
                (self.phone_hash(phone), since),
            ).fetchone()
        return row[0] if row else None

    def alert_exists(self, ref: str) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM alerts WHERE ref=?", (ref,)).fetchone() is not None

    def record_label(self, ref: str, label: int, note: str, phone: str, now: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO labels (ref, label, note, phone_hash, received_at) VALUES (?,?,?,?,?)",
                (ref, label, note[:140], self.phone_hash(phone), now),
            )
            self._conn.commit()

    def get_lang(self, phone: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT lang FROM prefs WHERE phone_hash=?", (self.phone_hash(phone),)
            ).fetchone()
        return row[0] if row else None

    def set_lang(self, phone: str, lang: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO prefs (phone_hash, lang) VALUES (?,?)",
                (self.phone_hash(phone), lang),
            )
            self._conn.commit()

    def record_inbound(self, phone: str, kind: str, now: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO inbound (phone_hash, kind, received_at) VALUES (?,?,?)",
                (self.phone_hash(phone), kind, now),
            )
            self._conn.commit()

    def inbound_count(self, phone: str, kind: str, since: float) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM inbound WHERE phone_hash=? AND kind=? AND received_at>=?",
                (self.phone_hash(phone), kind, since),
            ).fetchone()
        return int(row[0])

    def alerts_between(self, start: float, end: float) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ref, code, severity, asset_id, raised_at, cleared_at, values_json "
                "FROM alerts WHERE raised_at>=? AND raised_at<? ORDER BY raised_at",
                (start, end),
            ).fetchall()
        keys = ("ref", "code", "severity", "asset_id", "raised_at", "cleared_at", "values")
        return [dict(zip(keys, (*r[:6], json.loads(r[6])), strict=True)) for r in rows]

    def recent_alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        """Alerts with their technician labels, newest first (for the dashboard)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT a.ref, a.code, a.severity, a.asset_id, a.raised_at, a.cleared_at, "
                "(SELECT label FROM labels l WHERE l.ref=a.ref ORDER BY received_at DESC LIMIT 1) "
                "FROM alerts a ORDER BY a.raised_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("ref", "code", "severity", "asset_id", "raised_at", "cleared_at", "label")
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, value))
            self._conn.commit()

    def labels_by_code(self, since: float = 0.0) -> dict[str, list[int]]:
        """Latest label of each labelled alert, grouped by fault code, newest alert first.

        Only labels received after ``since`` count (see RESET TUNING).
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT a.code, (SELECT l.label FROM labels l WHERE l.ref=a.ref AND l.received_at>=? "
                "ORDER BY l.received_at DESC, l.id DESC LIMIT 1) AS label "
                "FROM alerts a ORDER BY a.raised_at DESC, a.rowid DESC",
                (since,),
            ).fetchall()
        out: dict[str, list[int]] = {}
        for code, label in rows:
            if label is not None:
                out.setdefault(code, []).append(int(label))
        return out

    def export_labelled(self) -> list[dict[str, Any]]:
        """Anonymised labelled alerts for the open fault dataset (no phone data)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT a.code, a.severity, a.raised_at, a.cleared_at, a.values_json, l.label "
                "FROM alerts a JOIN labels l ON l.ref=a.ref ORDER BY a.raised_at"
            ).fetchall()
        return [
            {"code": r[0], "severity": r[1], "raised_at": r[2], "cleared_at": r[3],
             "values": json.loads(r[4]), "label": r[5]}
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def now() -> float:
        return time.time()
