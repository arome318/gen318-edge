"""Power-quality excursion detector — pure Python state machine (frontier bet 6).

HONESTY CONSTRAINT (research brief ``bet-6-power-quality.md``, Q1/Q6): Gen318
telemetry is 30-900s polled; IEEE 1159's sag/swell categories require
sub-cycle RMS capture — its *shortest* category (instantaneous sag, 0.5 cycle
= 10ms @ 50Hz) is 3,000-90,000x faster than our poll rate, and even its
*slowest* short-duration category (temporary, up to 1min) still outruns a
900s poll. This module therefore NEVER emits "sag"/"swell"/any IEEE 1159 or
IEC 61000-4-30 Class A/S vocabulary. Every emitted event is one of the seven
honest, coarse `event_type`s below, carries `sample_period_seconds` and
`method="polled_window"`, and downstream consumers must not represent these
as compliance-grade PQ instrumentation. Cite: IEEE Std 1159 Table 4-2 (Perry
& Collins 2008), IEC 61000-4-30 (Fluke/EPRI summary).

THRESHOLD BAND — Ruling 11(b), the EXCURSION-EVENT band, NOT the narrower
NERC LV compliance band (216.2-243.8V / -6%/+6%) used for %-in-band
reporting in ``backend/app/services/power_quality/rollup.py``. This module
answers "is this bad enough to be alert-worthy"; rollup.py answers "are we
within the legally tight tolerance". Defaults (nominal 230V/50Hz,
undervoltage <0.90pu, overvoltage >1.10pu, frequency outside
[49.0, 51.0] Hz) are the research brief's NERC-derived bounds (Q3: NERC
Distribution Code Sec 4.3 LV table; Grid Code Sec 15.3.1 frequency). Severity
escalation at 0.85/1.15pu and NERC Grid Code's own "momentary excursions
tolerated to 51.5/47.5Hz" (Q3) are per the brief's guidance.

Mirrors ``backend/app/services/power_quality/detector.py`` exactly (same
constants, same state-machine semantics) — this file NEVER imports from
``backend/`` (edge containers do not ship that package); change both files
together or neither, per the ``adapters/base.py`` precedent.

No I/O of any kind happens here: no SQLite, no MQTT, no network. The
collector feeds ``Reading`` objects in; this returns plain dicts the caller
enqueues via ``EdgeDatabase.enqueue_sync``. Every threshold is env-overridable
so a site's local grid characteristics can be tuned without a firmware
rebuild.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from collector.adapters.base import Reading

# ── Constants (env-overridable; defaults per Ruling 11b) ──────────────────

NOMINAL_VOLTAGE_V = float(os.environ.get("PQ_NOMINAL_VOLTAGE_V", "230.0"))
NOMINAL_FREQUENCY_HZ = float(os.environ.get("PQ_NOMINAL_FREQUENCY_HZ", "50.0"))

UNDERVOLTAGE_PU = float(os.environ.get("PQ_UNDERVOLTAGE_PU", "0.90"))
OVERVOLTAGE_PU = float(os.environ.get("PQ_OVERVOLTAGE_PU", "1.10"))
INTERRUPTION_PU = float(os.environ.get("PQ_INTERRUPTION_PU", "0.10"))
SEVERITY_CRITICAL_UNDERVOLTAGE_PU = float(
    os.environ.get("PQ_SEVERITY_CRITICAL_UNDERVOLTAGE_PU", "0.85")
)
SEVERITY_CRITICAL_OVERVOLTAGE_PU = float(
    os.environ.get("PQ_SEVERITY_CRITICAL_OVERVOLTAGE_PU", "1.15")
)

FREQUENCY_LOW_HZ = float(os.environ.get("PQ_FREQUENCY_LOW_HZ", "49.0"))
FREQUENCY_HIGH_HZ = float(os.environ.get("PQ_FREQUENCY_HIGH_HZ", "51.0"))
# NERC Grid Code Sec 15.3.1 "momentary excursions tolerated to 51.5/47.5Hz"
# (research brief Q3). Severity escalation only — NOT the open/close band.
SEVERITY_CRITICAL_FREQUENCY_LOW_HZ = float(
    os.environ.get("PQ_SEVERITY_CRITICAL_FREQUENCY_LOW_HZ", "47.5")
)
SEVERITY_CRITICAL_FREQUENCY_HIGH_HZ = float(
    os.environ.get("PQ_SEVERITY_CRITICAL_FREQUENCY_HIGH_HZ", "51.5")
)

SEVERITY_CRITICAL_DURATION_SECONDS = float(
    os.environ.get("PQ_SEVERITY_CRITICAL_DURATION_SECONDS", "600.0")
)

# 2% of nominal, per the brief, for VOLTAGE. Frequency's own open band is
# already only 2% of nominal wide (49-51Hz around 50Hz), so applying the same
# fraction there would collapse the recovery band to a single point (~50.0Hz
# from both sides) — a fixed absolute margin is used instead. Same reasoning
# for the two ratio metrics (unbalance %, power factor), which have no
# physical "nominal" to take a fraction of. All four are documented judgment
# calls beyond the brief's literal "2% of nominal" text — see task report.
HYSTERESIS_FRACTION = float(os.environ.get("PQ_HYSTERESIS_FRACTION", "0.02"))
FREQUENCY_HYSTERESIS_HZ = float(os.environ.get("PQ_FREQUENCY_HYSTERESIS_HZ", "0.2"))
UNBALANCE_HYSTERESIS_PCT = float(os.environ.get("PQ_UNBALANCE_HYSTERESIS_PCT", "0.3"))
POWER_FACTOR_HYSTERESIS = float(os.environ.get("PQ_POWER_FACTOR_HYSTERESIS", "0.02"))

MIN_CONSECUTIVE_SAMPLES = int(os.environ.get("PQ_MIN_CONSECUTIVE_SAMPLES", "2"))
MIN_OPEN_DURATION_SECONDS = float(
    os.environ.get("PQ_MIN_OPEN_DURATION_SECONDS", "60.0")
)

UNBALANCE_THRESHOLD_PCT = float(os.environ.get("PQ_UNBALANCE_THRESHOLD_PCT", "3.0"))
# Common utility PF billing/penalty threshold (e.g. NERC-adjacent Disco
# tariff practice). The research brief gives NO numeric PF default — this is
# a documented judgment call, not sourced from bet-6-power-quality.md.
POWER_FACTOR_LOW_THRESHOLD = float(
    os.environ.get("PQ_POWER_FACTOR_LOW_THRESHOLD", "0.85")
)

METHOD = "polled_window"

# ── Metric name vocabulary ──────────────────────────────────────────────
# Literal duplication of the relevant backend/app/adapters/metrics.py names.
# Edge NEVER imports backend/ — keep this list in step with that file by
# hand, same discipline as adapters/base.py.

_VOLTAGE_METRICS: frozenset[str] = frozenset(
    {
        "meter.voltage_v",
        "inverter.ac_output_voltage_v",
        "inverter.ac_input_voltage_v",
        "grid.voltage_v",
        "genset.output_voltage_v",
    }
)
_PHASE_VOLTAGE_METRICS: dict[str, str] = {
    "meter.l1_voltage_v": "l1",
    "meter.l2_voltage_v": "l2",
    "meter.l3_voltage_v": "l3",
    "inverter.ac_output_l1_voltage_v": "l1",
    "inverter.ac_output_l2_voltage_v": "l2",
    "inverter.ac_output_l3_voltage_v": "l3",
}
_FREQUENCY_METRICS: frozenset[str] = frozenset(
    {
        "inverter.ac_output_freq_hz",
        "inverter.ac_input_freq_hz",
        "grid.frequency_hz",
        "genset.output_frequency_hz",
    }
)
_POWER_FACTOR_METRICS: frozenset[str] = frozenset({"meter.power_factor"})

# Each 3-tuple is one asset-scoped phase group for the unbalance formula —
# all three metrics must land in the SAME reading batch (one poll) to combine.
_UNBALANCE_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("meter.l1_voltage_v", "meter.l2_voltage_v", "meter.l3_voltage_v"),
    (
        "inverter.ac_output_l1_voltage_v",
        "inverter.ac_output_l2_voltage_v",
        "inverter.ac_output_l3_voltage_v",
    ),
)

EventType = Literal[
    "undervoltage",
    "overvoltage",
    "underfrequency",
    "overfrequency",
    "voltage_unbalance",
    "supply_interruption",
    "power_factor_low",
]
Zone = Literal[
    "normal",
    "undervoltage",
    "overvoltage",
    "interruption",
    "underfrequency",
    "overfrequency",
    "voltage_unbalance",
    "power_factor_low",
]

_ZONE_TO_EVENT_TYPE: dict[str, EventType] = {
    "undervoltage": "undervoltage",
    "overvoltage": "overvoltage",
    "interruption": "supply_interruption",
    "underfrequency": "underfrequency",
    "overfrequency": "overfrequency",
    "voltage_unbalance": "voltage_unbalance",
    "power_factor_low": "power_factor_low",
}

# Zones whose "worst" reading is the MINIMUM observed value; all others track
# the MAXIMUM. Used to accumulate the event's `magnitude` while open.
_UNDER_ZONES: frozenset[str] = frozenset(
    {"undervoltage", "interruption", "underfrequency", "power_factor_low"}
)

# Fix round 1 / IMPORTANT-2: a worsening condition (e.g. undervoltage
# collapsing into a full supply interruption) must escalate, not sit
# swallowed in the hysteresis-dead-zone branch waiting for the ORIGINAL,
# now-superseded zone's own recovery criteria — which it might never meet.
# Only "interruption" currently outranks anything (a strictly more severe
# reading within the same low-voltage family); every other zone has nothing
# worse defined above it yet. See _is_escalation() and its use in _advance().
_ZONE_SEVERITY_RANK: dict[Zone, int] = {
    "normal": 0,
    "undervoltage": 1,
    "overvoltage": 1,
    "underfrequency": 1,
    "overfrequency": 1,
    "voltage_unbalance": 1,
    "power_factor_low": 1,
    "interruption": 2,
}


def _is_escalation(new_zone: Zone, open_zone: Zone) -> bool:
    """True when ``new_zone`` is strictly more severe than the open zone."""
    return _ZONE_SEVERITY_RANK.get(new_zone, 0) > _ZONE_SEVERITY_RANK.get(open_zone, 0)


# _OPEN_ENQUEUE_TYPES gates OPEN-time enqueueing ONLY: only supply_interruption
# enqueues at OPEN (priority 0), because the DB-side idempotency key is
# INSERT ... ON CONFLICT DO NOTHING (no UPDATE-on-conflict), so an operator
# must find out immediately rather than wait for a close that may never come
# (supply may stay down for hours, or the edge may never reconnect).
#
# Fix round 1 / CRITICAL-1: CLOSE-time event building is now UNCONDITIONAL
# for every event_type, including supply_interruption (see _advance()) —
# the state machine itself no longer special-cases interruption at close.
# What stays "edge behavior as ruled" (open-only, priority 0, no second
# queue item for interruption) is enforced ONE layer up, in
# CollectorService._detect_power_quality (edge/collector/main.py): it drops
# any supply_interruption event whose ended_at is not None before
# enqueueing. The cloud task has no such filter — its own open/close
# bookkeeping is a direct UPDATE, not a queue insert, so it DOES act on the
# interruption close event; see backend/app/tasks/pq_tasks.py.
_OPEN_ENQUEUE_TYPES: frozenset[str] = frozenset({"supply_interruption"})


def compute_nema_unbalance_pct(v1: float, v2: float, v3: float) -> float:
    """NEMA/LVUR %VU: 100 x max deviation from the mean / mean.

    Magnitude-only approximation (research brief Q4, Pillay & Manyage IEEE
    PER 2001): NEMA systematically under-reads vs the true (symmetrical-
    component) VUF, worse as unbalance grows. This is that approximation,
    not the IEC "true" definition — callers must label it as such.
    """
    mean = (v1 + v2 + v3) / 3.0
    if mean == 0:
        return 0.0
    max_dev = max(abs(v1 - mean), abs(v2 - mean), abs(v3 - mean))
    return 100.0 * max_dev / mean


def _voltage_zone(value: float) -> Zone:
    if value < INTERRUPTION_PU * NOMINAL_VOLTAGE_V:
        return "interruption"
    if value < UNDERVOLTAGE_PU * NOMINAL_VOLTAGE_V:
        return "undervoltage"
    if value > OVERVOLTAGE_PU * NOMINAL_VOLTAGE_V:
        return "overvoltage"
    return "normal"


def _frequency_zone(value: float) -> Zone:
    if value < FREQUENCY_LOW_HZ:
        return "underfrequency"
    if value > FREQUENCY_HIGH_HZ:
        return "overfrequency"
    return "normal"


def _power_factor_zone(value: float) -> Zone:
    return "power_factor_low" if abs(value) < POWER_FACTOR_LOW_THRESHOLD else "normal"


def _unbalance_zone(pct: float) -> Zone:
    return "voltage_unbalance" if pct > UNBALANCE_THRESHOLD_PCT else "normal"


def _recovered(zone: Zone, value: float) -> bool:
    """True once ``value`` has crossed back past the TIGHTER inner recovery band."""
    if zone == "undervoltage":
        return value >= NOMINAL_VOLTAGE_V * (UNDERVOLTAGE_PU + HYSTERESIS_FRACTION)
    if zone == "overvoltage":
        return value <= NOMINAL_VOLTAGE_V * (OVERVOLTAGE_PU - HYSTERESIS_FRACTION)
    if zone == "interruption":
        return value >= NOMINAL_VOLTAGE_V * (INTERRUPTION_PU + HYSTERESIS_FRACTION)
    if zone == "underfrequency":
        return value >= FREQUENCY_LOW_HZ + FREQUENCY_HYSTERESIS_HZ
    if zone == "overfrequency":
        return value <= FREQUENCY_HIGH_HZ - FREQUENCY_HYSTERESIS_HZ
    if zone == "power_factor_low":
        return abs(value) >= POWER_FACTOR_LOW_THRESHOLD + POWER_FACTOR_HYSTERESIS
    if zone == "voltage_unbalance":
        return value <= UNBALANCE_THRESHOLD_PCT - UNBALANCE_HYSTERESIS_PCT
    return True


def _threshold_for(zone: Zone) -> float:
    return {
        "undervoltage": UNDERVOLTAGE_PU * NOMINAL_VOLTAGE_V,
        "overvoltage": OVERVOLTAGE_PU * NOMINAL_VOLTAGE_V,
        "interruption": INTERRUPTION_PU * NOMINAL_VOLTAGE_V,
        "underfrequency": FREQUENCY_LOW_HZ,
        "overfrequency": FREQUENCY_HIGH_HZ,
        "power_factor_low": POWER_FACTOR_LOW_THRESHOLD,
        "voltage_unbalance": UNBALANCE_THRESHOLD_PCT,
    }[zone]


def _nominal_for(zone: Zone) -> float | None:
    if zone in ("undervoltage", "overvoltage", "interruption"):
        return NOMINAL_VOLTAGE_V
    if zone in ("underfrequency", "overfrequency"):
        return NOMINAL_FREQUENCY_HZ
    return None  # power factor / unbalance are ratios, not physical quantities


def _classify_severity(
    event_type: str, magnitude: float, duration_seconds: float | None
) -> str:
    if event_type == "supply_interruption":
        return "critical"
    if (
        duration_seconds is not None
        and duration_seconds > SEVERITY_CRITICAL_DURATION_SECONDS
    ):
        return "critical"
    if event_type == "undervoltage" and magnitude < (
        SEVERITY_CRITICAL_UNDERVOLTAGE_PU * NOMINAL_VOLTAGE_V
    ):
        return "critical"
    if event_type == "overvoltage" and magnitude > (
        SEVERITY_CRITICAL_OVERVOLTAGE_PU * NOMINAL_VOLTAGE_V
    ):
        return "critical"
    if (
        event_type == "underfrequency"
        and magnitude < SEVERITY_CRITICAL_FREQUENCY_LOW_HZ
    ):
        return "critical"
    if (
        event_type == "overfrequency"
        and magnitude > SEVERITY_CRITICAL_FREQUENCY_HIGH_HZ
    ):
        return "critical"
    return "warning"


@dataclass
class _ChannelState:
    """Per-(asset_id, channel) open/pending tracking."""

    pending_zone: Zone | None = None
    pending_since: datetime | None = None
    pending_count: int = 0

    open_zone: Zone | None = None
    open_started_at: datetime | None = None
    open_extreme: float | None = None
    open_metric_name: str = ""
    open_sample_period_seconds: float | None = None
    recovery_count: int = 0


@dataclass
class PQDetector:
    """Pure state machine: feed readings, get back queue-ready event dicts.

    No I/O, no SQLite, no cloud awareness — the caller (CollectorService)
    enqueues whatever this returns via ``EdgeDatabase.enqueue_sync``, popping
    the ``"priority"`` key off each returned dict first.
    """

    _channels: dict[tuple[str, str], _ChannelState] = field(default_factory=dict)
    _last_seen_at: dict[tuple[str, str], datetime] = field(default_factory=dict)
    # Per-(asset_id, group) buffer of phase voltages awaiting a complete
    # l1/l2/l3 triplet from the SAME poll for the unbalance formula.
    _phase_buffer: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)

    def process_batch(self, readings: list[Reading]) -> list[dict[str, Any]]:
        """Feed one adapter poll's readings; return emitted pq_event payloads."""
        events: list[dict[str, Any]] = []
        for reading in readings:
            events.extend(self._process_phase_buffer(reading))
        for reading in readings:
            events.extend(self.process_reading(reading))
        return events

    def _process_phase_buffer(self, reading: Reading) -> list[dict[str, Any]]:
        phase = _PHASE_VOLTAGE_METRICS.get(reading.metric_name)
        if phase is None:
            return []
        asset_id = reading.asset_id or ""
        for group in _UNBALANCE_GROUPS:
            if reading.metric_name not in group:
                continue
            key = (asset_id, group[0])
            bucket = self._phase_buffer.setdefault(key, {})
            bucket[reading.metric_name] = reading.value
            if all(name in bucket for name in group):
                v1, v2, v3 = (bucket[name] for name in group)
                pct = compute_nema_unbalance_pct(v1, v2, v3)
                self._phase_buffer.pop(key, None)
                channel_key = (asset_id, "voltage_unbalance")
                last_seen = self._last_seen_at.get(channel_key)
                sample_period = (
                    (reading.timestamp - last_seen).total_seconds()
                    if last_seen is not None
                    else None
                )
                self._last_seen_at[channel_key] = reading.timestamp
                return self._advance(
                    channel_key=channel_key,
                    metric_name=group[0],
                    asset_id=asset_id,
                    value=pct,
                    timestamp=reading.timestamp,
                    zone=_unbalance_zone(pct),
                    phase=None,
                    sample_period_seconds=sample_period,
                )
        return []

    def process_reading(self, reading: Reading) -> list[dict[str, Any]]:
        """Feed a single reading; returns emitted pq_event payloads (usually [])."""
        metric = reading.metric_name
        asset_id = reading.asset_id or ""

        if metric in _VOLTAGE_METRICS:
            zone = _voltage_zone(reading.value)
            phase = None
        elif metric in _PHASE_VOLTAGE_METRICS:
            zone = _voltage_zone(reading.value)
            phase = _PHASE_VOLTAGE_METRICS[metric]
        elif metric in _FREQUENCY_METRICS:
            zone = _frequency_zone(reading.value)
            phase = None
        elif metric in _POWER_FACTOR_METRICS:
            zone = _power_factor_zone(reading.value)
            phase = None
        else:
            return []  # not a metric this detector tracks — zero cost

        channel_key = (asset_id, metric)
        last_seen = self._last_seen_at.get(channel_key)
        sample_period = (
            (reading.timestamp - last_seen).total_seconds()
            if last_seen is not None
            else None
        )
        self._last_seen_at[channel_key] = reading.timestamp

        return self._advance(
            channel_key=channel_key,
            metric_name=metric,
            asset_id=asset_id,
            value=reading.value,
            timestamp=reading.timestamp,
            zone=zone,
            phase=phase,
            sample_period_seconds=sample_period,
        )

    def _advance(
        self,
        *,
        channel_key: tuple[str, str],
        metric_name: str,
        asset_id: str,
        value: float,
        timestamp: datetime,
        zone: Zone,
        phase: str | None,
        sample_period_seconds: float | None,
    ) -> list[dict[str, Any]]:
        state = self._channels.setdefault(channel_key, _ChannelState())
        events: list[dict[str, Any]] = []

        if state.open_zone is None:
            if zone == "normal":
                state.pending_zone = None
                state.pending_count = 0
                state.pending_since = None
                return events
            if state.pending_zone != zone:
                state.pending_zone = zone
                state.pending_since = timestamp
                state.pending_count = 1
                return events
            state.pending_count += 1
            elapsed = (
                (timestamp - state.pending_since).total_seconds()
                if state.pending_since is not None
                else 0.0
            )
            if (
                state.pending_count >= MIN_CONSECUTIVE_SAMPLES
                and elapsed >= MIN_OPEN_DURATION_SECONDS
            ):
                opened_at = state.pending_since
                state.open_zone = zone
                state.open_started_at = opened_at
                state.open_extreme = value
                state.open_metric_name = metric_name
                state.open_sample_period_seconds = sample_period_seconds
                state.recovery_count = 0
                state.pending_zone = None
                state.pending_count = 0
                state.pending_since = None
                if _ZONE_TO_EVENT_TYPE[zone] in _OPEN_ENQUEUE_TYPES:
                    events.append(
                        self._build_event(
                            zone=zone,
                            asset_id=asset_id,
                            phase=phase,
                            metric_name=metric_name,
                            magnitude=value,
                            started_at=opened_at,
                            ended_at=None,
                            sample_period_seconds=sample_period_seconds,
                        )
                    )
            return events

        # An excursion is currently open on this channel.
        if zone == state.open_zone:
            state.recovery_count = 0
            extreme = state.open_extreme if state.open_extreme is not None else value
            state.open_extreme = (
                min(extreme, value) if zone in _UNDER_ZONES else max(extreme, value)
            )
            return events

        if _is_escalation(zone, state.open_zone):
            # Fix round 1 / IMPORTANT-2: close-and-supersede IMMEDIATELY, on
            # the first sample that lands in the strictly-worse zone — do
            # NOT wait for MIN_CONSECUTIVE_SAMPLES of escalation the way
            # recovery does. The new (worse) zone still has to earn its own
            # open through the normal >=2-consecutive-samples/>=60s gate via
            # the recursive call below (a fresh _ChannelState starts
            # pending, not open) — escalation only removes the OLD, now-
            # superseded zone's grip on the channel, it does not skip the
            # new zone's own confirmation.
            events.append(
                self._build_event(
                    zone=state.open_zone,
                    asset_id=asset_id,
                    phase=phase,
                    metric_name=state.open_metric_name,
                    magnitude=(
                        state.open_extreme if state.open_extreme is not None else value
                    ),
                    started_at=state.open_started_at,
                    ended_at=timestamp,
                    sample_period_seconds=state.open_sample_period_seconds,
                )
            )
            self._channels[channel_key] = _ChannelState()
            events.extend(
                self._advance(
                    channel_key=channel_key,
                    metric_name=metric_name,
                    asset_id=asset_id,
                    value=value,
                    timestamp=timestamp,
                    zone=zone,
                    phase=phase,
                    sample_period_seconds=sample_period_seconds,
                )
            )
            return events

        if _recovered(state.open_zone, value):
            state.recovery_count += 1
            if state.recovery_count >= MIN_CONSECUTIVE_SAMPLES:
                closed_zone = state.open_zone
                # Fix round 1 / CRITICAL-1: CLOSE always builds+appends the
                # event now, for every event_type including
                # supply_interruption — see _OPEN_ENQUEUE_TYPES's comment.
                events.append(
                    self._build_event(
                        zone=closed_zone,
                        asset_id=asset_id,
                        phase=phase,
                        metric_name=state.open_metric_name,
                        magnitude=(
                            state.open_extreme
                            if state.open_extreme is not None
                            else value
                        ),
                        started_at=state.open_started_at,
                        ended_at=timestamp,
                        sample_period_seconds=state.open_sample_period_seconds,
                    )
                )
                self._channels[channel_key] = _ChannelState()
                # Re-evaluate this same sample against the now-fresh channel
                # state: it may itself start a NEW, different-zone excursion
                # (e.g. an interruption recovering straight into undervoltage).
                events.extend(
                    self._advance(
                        channel_key=channel_key,
                        metric_name=metric_name,
                        asset_id=asset_id,
                        value=value,
                        timestamp=timestamp,
                        zone=zone,
                        phase=phase,
                        sample_period_seconds=sample_period_seconds,
                    )
                )
            return events

        # Hysteresis dead zone (or a not-yet-confirmed different OOB zone
        # that isn't a strict escalation either — e.g. a partial recovery
        # blip): stays open, doesn't count toward recovery, doesn't extend
        # extremes.
        state.recovery_count = 0
        return events

    def _build_event(
        self,
        *,
        zone: Zone,
        asset_id: str,
        phase: str | None,
        metric_name: str,
        magnitude: float,
        started_at: datetime | None,
        ended_at: datetime | None,
        sample_period_seconds: float | None,
    ) -> dict[str, Any]:
        event_type = _ZONE_TO_EVENT_TYPE[zone]
        duration_seconds = (
            (ended_at - started_at).total_seconds()
            if started_at is not None and ended_at is not None
            else None
        )
        severity = _classify_severity(event_type, magnitude, duration_seconds)
        threshold = _threshold_for(zone)
        return {
            "type": "pq_event",
            "event_type": event_type,
            "asset_id": asset_id or None,
            "phase": phase,
            "magnitude": magnitude,
            "threshold": threshold,
            "nominal": _nominal_for(zone),
            "started_at": started_at.isoformat() if started_at else None,
            "ended_at": ended_at.isoformat() if ended_at else None,
            "duration_seconds": duration_seconds,
            "sample_period_seconds": sample_period_seconds,
            "severity": severity,
            "method": METHOD,
            # priority: consumed and popped by the collector before enqueueing
            # — never part of the pq_events row itself. OPEN-only concept
            # (urgent, fire-immediately); a close event (ended_at is not
            # None) is always 1, even for supply_interruption — the
            # collector additionally drops interruption CLOSE events
            # entirely rather than enqueueing them (see main.py).
            "priority": 0
            if (ended_at is None and event_type in _OPEN_ENQUEUE_TYPES)
            else 1,
            # Best-effort context so the existing alerts.critical pipeline
            # (CriticalAlert schema, app/api/v1/ingest.py::ingest_alert) shows
            # a meaningful alert on the priority-0 interruption path, which
            # cannot carry the full pq_event payload through. Harmless extra
            # keys on the priority-1 (events.raw) path.
            "metric_name": metric_name,
            "metric_value": magnitude,
            "threshold_value": threshold,
            "title": f"{event_type} on {metric_name}",
        }
