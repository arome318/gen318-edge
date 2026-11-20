"""Tests for the edge power-quality excursion detector (pq_detector.py).

Pure state-machine tests — no SQLite/MQTT/network involved except
``test_offline_queues_locally``, which proves the detector's output can be
queued through the real ``EdgeDatabase`` with zero network I/O.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from collector.adapters.base import Reading
from collector.pq_detector import (
    FREQUENCY_HIGH_HZ,
    FREQUENCY_LOW_HZ,
    NOMINAL_VOLTAGE_V,
    UNBALANCE_THRESHOLD_PCT,
    UNDERVOLTAGE_PU,
    PQDetector,
    compute_nema_unbalance_pct,
)

ASSET_ID = "11111111-1111-1111-1111-111111111111"
T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _reading(
    metric: str, value: float, at: datetime, asset_id: str = ASSET_ID
) -> Reading:
    return Reading(metric_name=metric, value=value, timestamp=at, asset_id=asset_id)


def test_excursion_opens_after_two_samples_and_min_duration() -> None:
    """No event until >=2 consecutive out-of-band samples spanning >=60s."""
    detector = PQDetector()

    # First out-of-band sample: nothing opens yet.
    events = detector.process_reading(_reading("meter.voltage_v", 190.0, T0))
    assert events == []

    # Second sample only 30s later: count satisfied, span not yet (<60s).
    events = detector.process_reading(
        _reading("meter.voltage_v", 190.0, T0 + timedelta(seconds=30))
    )
    assert events == []

    # Third sample at 60s total span: both conditions now satisfied, but
    # opening alone never enqueues for undervoltage — it enqueues on CLOSE.
    events = detector.process_reading(
        _reading("meter.voltage_v", 190.0, T0 + timedelta(seconds=60))
    )
    assert events == []

    # Recovery: 2 consecutive samples past the tighter inner recovery band
    # closes the excursion and emits exactly one event.
    recovery_v = NOMINAL_VOLTAGE_V * (UNDERVOLTAGE_PU + 0.02) + 1.0
    detector.process_reading(
        _reading("meter.voltage_v", recovery_v, T0 + timedelta(seconds=90))
    )
    events = detector.process_reading(
        _reading("meter.voltage_v", recovery_v, T0 + timedelta(seconds=120))
    )

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "pq_event"
    assert event["event_type"] == "undervoltage"
    assert event["severity"] in {"warning", "critical"}
    assert event["magnitude"] == 190.0
    assert event["started_at"] == T0.isoformat()
    assert event["ended_at"] == (T0 + timedelta(seconds=120)).isoformat()
    assert event["duration_seconds"] == 120.0
    assert event["method"] == "polled_window"
    assert event["priority"] == 1


def test_hysteresis_prevents_flapping() -> None:
    """Oscillating just above the open threshold (but inside the hysteresis
    dead zone) must never close the excursion or emit extra events."""
    detector = PQDetector()

    # Open an undervoltage excursion.
    for offset in (0, 30, 60):
        detector.process_reading(
            _reading("meter.voltage_v", 190.0, T0 + timedelta(seconds=offset))
        )

    # Dead-zone value: back inside the open band (>=207V) but below the
    # tighter recovery threshold (~211.6V) — must NOT close the event.
    dead_zone_v = NOMINAL_VOLTAGE_V * UNDERVOLTAGE_PU + 1.0
    events: list[object] = []
    t = T0 + timedelta(seconds=90)
    for _ in range(6):
        events += detector.process_reading(_reading("meter.voltage_v", dead_zone_v, t))
        t += timedelta(seconds=30)
        events += detector.process_reading(_reading("meter.voltage_v", 190.0, t))
        t += timedelta(seconds=30)

    assert events == [], "oscillation inside the hysteresis band must not flap"


def test_interruption_priority_zero() -> None:
    """supply_interruption enqueues immediately on OPEN at priority 0."""
    detector = PQDetector()

    events = detector.process_reading(_reading("meter.voltage_v", 5.0, T0))
    assert events == []
    events = detector.process_reading(
        _reading("meter.voltage_v", 5.0, T0 + timedelta(seconds=30))
    )
    assert events == []
    events = detector.process_reading(
        _reading("meter.voltage_v", 5.0, T0 + timedelta(seconds=60))
    )

    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "supply_interruption"
    assert event["priority"] == 0
    assert event["severity"] == "critical"
    assert event["ended_at"] is None
    assert event["started_at"] == T0.isoformat()


def test_offline_queues_locally(edge_db) -> None:  # type: ignore[no-untyped-def]
    """Detector output queues through the real SQLite EdgeDatabase — zero
    network calls are made anywhere in this path."""
    detector = PQDetector()
    events: list[dict[str, object]] = []
    for offset in (0, 30, 60):
        events = detector.process_reading(
            _reading("meter.voltage_v", 5.0, T0 + timedelta(seconds=offset))
        )
    # The third sample (offset=60) both satisfies count+span AND is the
    # interruption zone, so the open fires here.
    assert len(events) == 1

    for event in events:
        priority = event.pop("priority")
        edge_db.enqueue_sync(priority=priority, payload=event)

    queued = edge_db.get_queue_items(priority=0)
    assert len(queued) == 1
    assert queued[0]["payload"]["type"] == "pq_event"
    assert queued[0]["payload"]["event_type"] == "supply_interruption"


def test_unbalance_formula() -> None:
    """NEMA %VU = 100 * max deviation from mean / mean (magnitude-only)."""
    # Balanced three-phase system: 0% unbalance.
    assert compute_nema_unbalance_pct(230.0, 230.0, 230.0) == 0.0

    # Worked example structurally consistent with the research brief's
    # NEMA-under-reads-true-VUF table (Q4): a clearly unbalanced triplet
    # must read comfortably above the 3% flag threshold.
    pct = compute_nema_unbalance_pct(220.0, 230.0, 240.0)
    mean = (220.0 + 230.0 + 240.0) / 3.0
    expected = (
        100.0 * max(abs(220.0 - mean), abs(230.0 - mean), abs(240.0 - mean)) / mean
    )
    assert pct == pytest.approx(expected)
    assert pct > UNBALANCE_THRESHOLD_PCT

    # Zero mean must not raise (degenerate guard).
    assert compute_nema_unbalance_pct(0.0, 0.0, 0.0) == 0.0


def test_unbalance_detected_from_phase_triplet_batch() -> None:
    """Per-phase voltages arriving in one batch combine into one unbalance
    channel — event opens/closes exactly like any other excursion."""
    detector = PQDetector()
    t = T0
    for _ in range(3):
        # Individually each phase stays within the normal voltage band
        # (207-253V) — only the AGGREGATE unbalance (~8.7%) crosses 3%, so
        # this isolates the unbalance channel from the per-phase channels.
        readings = [
            _reading("meter.l1_voltage_v", 210.0, t),
            _reading("meter.l2_voltage_v", 230.0, t),
            _reading("meter.l3_voltage_v", 250.0, t),
        ]
        events = detector.process_batch(readings)
        t += timedelta(seconds=30)

    # By the 3rd batch (60s span, 3 samples) the unbalance excursion is open
    # (no event yet — it fires on close).
    assert events == []

    recovered = [
        _reading("meter.l1_voltage_v", 230.0, t),
        _reading("meter.l2_voltage_v", 230.0, t),
        _reading("meter.l3_voltage_v", 230.0, t),
    ]
    detector.process_batch(recovered)
    t += timedelta(seconds=30)
    events = detector.process_batch(
        [
            _reading("meter.l1_voltage_v", 230.0, t),
            _reading("meter.l2_voltage_v", 230.0, t),
            _reading("meter.l3_voltage_v", 230.0, t),
        ]
    )

    assert len(events) == 1
    assert events[0]["event_type"] == "voltage_unbalance"
    assert events[0]["phase"] is None


def test_frequency_excursion_uses_frequency_band() -> None:
    detector = PQDetector()
    low = FREQUENCY_LOW_HZ - 0.5
    for offset in (0, 30, 60):
        detector.process_reading(
            _reading("inverter.ac_output_freq_hz", low, T0 + timedelta(seconds=offset))
        )
    recovered = FREQUENCY_HIGH_HZ  # comfortably past the recovery margin below
    recovered = 50.0
    detector.process_reading(
        _reading("inverter.ac_output_freq_hz", recovered, T0 + timedelta(seconds=90))
    )
    events = detector.process_reading(
        _reading("inverter.ac_output_freq_hz", recovered, T0 + timedelta(seconds=120))
    )
    assert len(events) == 1
    assert events[0]["event_type"] == "underfrequency"
    assert events[0]["nominal"] == 50.0


def test_irrelevant_metric_ignored() -> None:
    detector = PQDetector()
    events = detector.process_reading(_reading("battery.soc_pct", 55.0, T0))
    assert events == []


# ── Fix round 1 ──────────────────────────────────────────────────────────


def test_interruption_close_event_is_built_but_not_enqueued_by_collector() -> None:
    """CRITICAL-1 (detector half): the shared state machine now ALWAYS
    builds a close event for supply_interruption too — process_batch()
    itself must return it. (The collector-level filter that drops it before
    enqueueing is a main.py concern, exercised in edge/tests/test_collector_*
    integration coverage, not here.)"""
    detector = PQDetector()
    for offset in (0, 30, 60):
        detector.process_reading(
            _reading("meter.voltage_v", 5.0, T0 + timedelta(seconds=offset))
        )

    recovery_v = (
        NOMINAL_VOLTAGE_V * 0.5
    )  # comfortably past the interruption recovery margin
    t = T0 + timedelta(seconds=90)
    detector.process_reading(_reading("meter.voltage_v", recovery_v, t))
    t += timedelta(seconds=30)
    events = detector.process_reading(_reading("meter.voltage_v", recovery_v, t))

    assert len(events) == 1
    close_event = events[0]
    assert close_event["event_type"] == "supply_interruption"
    assert close_event["severity"] == "critical"
    assert close_event["ended_at"] == t.isoformat()
    assert close_event["started_at"] == T0.isoformat()
    assert close_event["duration_seconds"] == (t - T0).total_seconds()
    assert close_event["priority"] == 1  # OPEN-only priority-0; close is always 1


def test_worsening_undervoltage_escalates_to_interruption_priority_zero() -> None:
    """IMPORTANT-2: 220V (normal) -> 200V x2 (undervoltage opens) -> 15V x2
    (escalates: undervoltage force-closes, interruption opens at P0)."""
    detector = PQDetector()
    t = T0

    assert detector.process_reading(_reading("meter.voltage_v", 220.0, t)) == []
    t += timedelta(seconds=30)

    assert detector.process_reading(_reading("meter.voltage_v", 200.0, t)) == []
    t += timedelta(seconds=60)
    events = detector.process_reading(_reading("meter.voltage_v", 200.0, t))
    assert events == []  # undervoltage opens silently (only interruption fires at open)

    t += timedelta(seconds=30)
    escalation_events = detector.process_reading(_reading("meter.voltage_v", 15.0, t))
    assert len(escalation_events) == 1
    assert escalation_events[0]["event_type"] == "undervoltage"
    assert (
        escalation_events[0]["ended_at"] == t.isoformat()
    )  # force-closed by escalation

    t += timedelta(seconds=60)
    events = detector.process_reading(_reading("meter.voltage_v", 15.0, t))

    assert len(events) == 1
    interruption_event = events[0]
    assert interruption_event["event_type"] == "supply_interruption"
    assert interruption_event["priority"] == 0
    assert interruption_event["severity"] == "critical"
    assert interruption_event["ended_at"] is None
