from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from common.storage import (
    StoragePolicy,
    StoragePressure,
    assess_storage,
    database_footprint_bytes,
)


def test_estimated_daily_bytes_uses_declared_fleet_shape() -> None:
    policy = StoragePolicy(
        expected_assets=2,
        metrics_per_asset=3,
        bytes_per_measurement=100,
        poll_interval_seconds=60,
    )
    assert policy.estimated_daily_bytes == 864_000


@pytest.mark.parametrize(
    ("used", "free", "expected"),
    [
        (10, 90, StoragePressure.NORMAL),
        (81, 19, StoragePressure.WARNING),
        (91, 9, StoragePressure.CRITICAL),
        (97, 3, StoragePressure.EMERGENCY),
    ],
)
def test_assess_storage_pressure_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    used: int,
    free: int,
    expected: StoragePressure,
) -> None:
    monkeypatch.setattr(
        "common.storage.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=100, used=used, free=free),
    )
    policy = StoragePolicy(
        target_days=1,
        expected_assets=1,
        metrics_per_asset=1,
        bytes_per_measurement=1,
        poll_interval_seconds=86_400,
        reserve_bytes=0,
    )
    result = assess_storage(str(tmp_path / "edge.db"), policy)
    assert result.pressure == expected


def test_reserve_can_trigger_emergency_before_percentage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "common.storage.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=1_000, used=100, free=900),
    )
    result = assess_storage(
        str(tmp_path / "edge.db"),
        StoragePolicy(reserve_bytes=900),
    )
    assert result.pressure == StoragePressure.EMERGENCY
    assert result.usable_free_bytes == 0


def test_database_footprint_includes_wal_and_shm(tmp_path: Path) -> None:
    database = tmp_path / "edge.db"
    database.write_bytes(b"db")
    (tmp_path / "edge.db-wal").write_bytes(b"wal")
    (tmp_path / "edge.db-shm").write_bytes(b"shm!")
    assert database_footprint_bytes(str(database)) == 9
