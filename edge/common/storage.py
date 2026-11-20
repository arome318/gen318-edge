"""Storage runway and pressure controls for offline-first edge nodes.

The policy is intentionally conservative: cloud-confirmed/rejected rows may be
pruned, but unsynced measurements are never selected for deletion. Runway is an
engineering estimate based on configured fleet shape and sampling rate, exposed
alongside actual filesystem capacity so hardware soak tests can validate it.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path


class StoragePressure(StrEnum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class StoragePolicy:
    target_days: int = 30
    expected_assets: int = 10
    metrics_per_asset: int = 40
    bytes_per_measurement: int = 160
    poll_interval_seconds: int = 60
    reserve_bytes: int = 2 * 1024**3
    warning_used_pct: float = 80.0
    critical_used_pct: float = 90.0
    emergency_used_pct: float = 96.0

    @property
    def estimated_daily_bytes(self) -> int:
        samples_per_day = 86_400 / max(self.poll_interval_seconds, 1)
        return max(
            1,
            int(
                self.expected_assets
                * self.metrics_per_asset
                * samples_per_day
                * self.bytes_per_measurement
            ),
        )


@dataclass(frozen=True)
class StorageStatus:
    pressure: StoragePressure
    filesystem_path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    usable_free_bytes: int
    used_pct: float
    database_bytes: int
    estimated_daily_bytes: int
    estimated_runway_days: float
    target_days: int
    target_met: bool
    reserve_bytes: int

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["pressure"] = self.pressure.value
        return data


def database_footprint_bytes(sqlite_path: str) -> int:
    """Return SQLite main/WAL/SHM footprint without failing on absent files."""
    total = 0
    for path in (sqlite_path, f"{sqlite_path}-wal", f"{sqlite_path}-shm"):
        try:
            total += os.path.getsize(path)
        except OSError:
            continue
    return total


def assess_storage(sqlite_path: str, policy: StoragePolicy) -> StorageStatus:
    """Assess filesystem pressure and estimated offline retention runway."""
    database_path = Path(sqlite_path)
    probe = database_path if database_path.exists() else database_path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent

    usage = shutil.disk_usage(probe)
    used_pct = (usage.used / usage.total * 100.0) if usage.total else 100.0
    usable_free = max(0, usage.free - policy.reserve_bytes)
    runway = usable_free / policy.estimated_daily_bytes

    if usage.free <= policy.reserve_bytes or used_pct >= policy.emergency_used_pct:
        pressure = StoragePressure.EMERGENCY
    elif used_pct >= policy.critical_used_pct or runway < 7:
        pressure = StoragePressure.CRITICAL
    elif used_pct >= policy.warning_used_pct or runway < policy.target_days:
        pressure = StoragePressure.WARNING
    else:
        pressure = StoragePressure.NORMAL

    return StorageStatus(
        pressure=pressure,
        filesystem_path=str(probe),
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        usable_free_bytes=usable_free,
        used_pct=round(used_pct, 2),
        database_bytes=database_footprint_bytes(sqlite_path),
        estimated_daily_bytes=policy.estimated_daily_bytes,
        estimated_runway_days=round(runway, 2),
        target_days=policy.target_days,
        target_met=runway >= policy.target_days,
        reserve_bytes=policy.reserve_bytes,
    )
