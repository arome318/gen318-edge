"""Cloud/edge value parity for battery health features.

The cloud TRAINS the battery-health model and the edge SERVES it, so a feature
name that means two different physical quantities on the two sides feeds
XGBoost inputs from a distribution it never saw. That failure is silent: the
vector has the right length, the right names and the right order.

`backend/tests/test_cloud_edge_feature_contract_unit.py` pins names and ORDER.
This pins VALUES. The scenario below is deliberately analytic — every expected
number is derivable by hand from the cloud formula in
`backend/app/ml/features/battery_features.py`, so this is a contract, not a
snapshot of whatever the edge happens to produce today.

Keep this file and its backend twin
(`tests/test_battery_features_unit.py::TestCloudEdgeValueParity`) in step: they
encode the same scenario and must agree on the same numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Shared analytic scenario ────────────────────────────────────────────
# 40 samples, 5-minute cadence. Pack voltage ramps 48 -> 56 V across the
# window (state of charge), and a 2.0 V load droop is applied to four samples
# in the MIDDLE of that ramp, where the ramp's own value is ~= its mean.
#
# This separates the two quantities that were conflated:
#   load droop      = 2.0 V   (what the cloud measures: current-conditioned)
#   full SoC swing  = 8.0 V   (what the edge measured: unconditioned spread)
# A test where voltage is constant cannot tell them apart — both return 2.0.
ASSET_ID = "11111111-1111-4111-8111-111111111111"
N = 40
CADENCE_MS = 5 * 60 * 1000
START_MS = 1_800_000_000_000
V_LOW, V_HIGH = 48.0, 56.0
DROOP_V = 2.0
LOAD_IDX = (18, 19, 20, 21)
# Rest current is 0 A so the edge's relaxation window (|I| < 0.1) is
# populated — otherwise relaxation_voltage_std is 0 and the SEI
# assertion below degenerates into 'equals zero', which any bug passes.
I_REST, I_LOAD = 0.0, 100.0

EXPECTED_VOLTAGE_SAG_V = 2.0
SOC_SWING_V = V_HIGH - V_LOW  # 8.0 — what a broken edge reports instead


def _scenario() -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    ts = [START_MS + i * CADENCE_MS for i in range(N)]
    ramp = np.linspace(V_LOW, V_HIGH, N)
    voltage = ramp.copy()
    current = np.full(N, I_REST)
    for i in LOAD_IDX:
        voltage[i] -= DROOP_V
        current[i] = I_LOAD
    temperature = np.full(N, 30.0)
    return ts, voltage, current, temperature


def _load(db, ts, voltage, current, temperature) -> None:
    batch: list[tuple[int, str, str, float, int]] = []
    for i, t in enumerate(ts):
        batch.append((t, ASSET_ID, "battery.voltage_v", float(voltage[i]), 1))
        batch.append((t, ASSET_ID, "battery.current_a", float(current[i]), 1))
        batch.append((t, ASSET_ID, "battery.temperature_c", float(temperature[i]), 1))
        batch.append((t, ASSET_ID, "battery.soc_pct", 50.0, 1))
    db.insert_measurements(batch)


def test_voltage_sag_measures_load_droop_not_soc_swing(edge_db) -> None:
    """voltage_sag_v must be current-conditioned, as the cloud computes it.

    An unconditioned percentile spread over all voltage reports the state-of-
    charge swing instead — here 8.0 V against a true load droop of 2.0 V.
    """
    from inference.features import build_battery_health_features

    ts, voltage, current, temperature = _scenario()
    _load(edge_db, ts, voltage, current, temperature)

    features = build_battery_health_features(edge_db, ASSET_ID)

    assert features["voltage_sag_v"] == pytest.approx(EXPECTED_VOLTAGE_SAG_V, abs=0.35)
    # Guard the specific regression: never the raw SoC swing.
    assert features["voltage_sag_v"] < SOC_SWING_V / 2


def test_sei_growth_proxy_uses_the_cloud_formula(edge_db) -> None:
    """sei_growth_proxy = relaxation_voltage_std / sqrt(age_days).

    The edge previously used voltage_sag * thermal_stress / age_days — a
    different formula entirely, under the same feature name.
    """
    from inference.features import build_battery_health_features

    ts, voltage, current, temperature = _scenario()
    _load(edge_db, ts, voltage, current, temperature)

    features = build_battery_health_features(edge_db, ASSET_ID)

    age_days = features["age_days"]
    assert age_days > 0
    expected = features["relaxation_voltage_std"] / (age_days**0.5)
    assert features["sei_growth_proxy"] == pytest.approx(expected, rel=1e-6)
