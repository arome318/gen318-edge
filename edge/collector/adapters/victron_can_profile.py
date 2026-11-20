"""The Victron-style CAN BMS profile, shared by BYD and the generic listener.

Victron's CAN-bus BMS protocol is what most inverter-compatible battery packs
speak: FreedomWon, Pylontech-class packs, BYD, and anything an installer sets
to "Battery Type 00" on a Deye or SunSynk. The frames are identical across
them; only the vendor extensions differ, which is why BYD's per-module cell
frames stay a BYD-only addition rather than part of this profile.

SOURCES (full ledger: docs/vendor-integrations/SOURCES.md)
  - This repo's own shipped ``byd_can.py``, which has been running these
    layouts. The field offsets, widths, scales and signedness below are
    byte-identical to it, so the BYD contract does not move.
  - ai-republic/bms-to-inverter @ cb12c01f and ArminJo/JK-BMSToPylontechCAN
    @ 05d767cd, read 18 September 2026 for the alarm frame and the frames
    this repo did not already implement.

A CORRECTION TO THE SPEC. The spec names 0x35A as the alarm frame. No source
binds an alarm frame to 0x35A: both implementations use 0x359, and one of
them carries a ``// 0x35A`` comment directly above a call that sends 0x359 —
which is the likely origin of the number. This profile uses 0x359.

AND A LIMIT ON WHAT IT CLAIMS. The two sources give DIFFERENT per-bit layouts
for 0x359. They agree only on the coarse fact that all-zero means no alarm
and any set bit in the first four bytes means the pack is flagging something.
That is all ``battery.state`` reports here. Picking one source's bit table
would turn "cell overvoltage" and "charge overcurrent" into confidently
labelled states that the other source says mean something else.
"""

from __future__ import annotations

import struct
from datetime import datetime
from typing import Any

import structlog

from .base import Reading

logger = structlog.get_logger()

#: Shared Victron-profile frames. Field layout is byte-identical to the
#: table ``byd_can.py`` has been shipping.
VICTRON_FRAMES: dict[int, dict[str, Any]] = {
    0x351: {
        "name": "charge_limits",
        "fields": {
            "battery.charge_voltage_v": {
                "offset": 0,
                "length": 2,
                "scale": 0.1,
                "signed": False,
            },
            "battery.charge_current_limit_a": {
                "offset": 2,
                "length": 2,
                "scale": 0.1,
                "signed": False,
            },
            "battery.discharge_current_limit_a": {
                "offset": 4,
                "length": 2,
                "scale": 0.1,
                "signed": False,
            },
        },
    },
    0x355: {
        "name": "soc_soh",
        "fields": {
            "battery.soc_pct": {
                "offset": 0,
                "length": 2,
                "scale": 1.0,
                "signed": False,
            },
            "battery.soh_pct": {
                "offset": 2,
                "length": 2,
                "scale": 1.0,
                "signed": False,
            },
        },
    },
    0x356: {
        "name": "voltage_current_temp",
        "fields": {
            "battery.voltage_v": {
                "offset": 0,
                "length": 2,
                "scale": 0.01,
                "signed": False,
            },
            "battery.current_a": {
                "offset": 2,
                "length": 2,
                "scale": 0.1,
                "signed": True,
            },
            "battery.temperature_c": {
                "offset": 4,
                "length": 2,
                "scale": 0.1,
                "signed": True,
            },
        },
    },
    0x35E: {
        "name": "manufacturer",
        "fields": {},  # ASCII identity only, never telemetry
    },
    0x35F: {
        "name": "model_info",
        "fields": {},  # model and firmware, identity only
    },
}

#: The alarm/warning frame. 0x359, not the spec's 0x35A.
ALARM_FRAME_ID = 0x359
#: Bytes 0-3 carry alarms and warnings in BOTH source layouts.
_ALARM_BYTES = 4

#: Frames carrying ASCII identity rather than telemetry.
IDENTITY_FRAMES = {0x35E: "manufacturer", 0x35F: "model"}

#: Profiles this module can decode. ``byd`` adds per-module cell frames on
#: top of the shared set; ``victron_generic`` is the shared set alone.
PROFILES = ("victron_generic", "byd")


def decode_fields(
    arb_id: int, data: bytes, asset_id: str, now: datetime
) -> list[Reading]:
    """Decode one shared-profile frame into readings.

    Returns an empty list for a frame this profile does not carry, and for a
    frame too short to hold the field it declares — a truncated frame is a bus
    problem, not a reading of zero.
    """
    frame = VICTRON_FRAMES.get(arb_id)
    if frame is None:
        return []

    readings: list[Reading] = []
    for metric_name, field in frame["fields"].items():
        offset: int = field["offset"]
        length: int = field["length"]
        raw_bytes = data[offset : offset + length]
        if len(raw_bytes) != length:
            continue
        if length == 2:
            fmt = "<h" if field["signed"] else "<H"
        elif length == 1:
            fmt = "<b" if field["signed"] else "<B"
        else:
            continue
        raw_value = struct.unpack(fmt, raw_bytes)[0]
        readings.append(
            Reading(
                metric_name=metric_name,
                value=raw_value * field["scale"],
                timestamp=now,
                quality=0,
                raw_register=f"can:0x{arb_id:03X}:{offset}",
                asset_id=asset_id,
            )
        )
    return readings


def decode_alarm(data: bytes, asset_id: str, now: datetime) -> list[Reading]:
    """Decode frame 0x359 into a COARSE battery state.

    0.0 = nothing flagged, 1.0 = the pack is flagging an alarm or warning.

    That is deliberately all it says. The two public implementations of this
    frame assign different meanings to the same bits, so a per-bit decode
    would publish one source's "cell overvoltage" where the other reads
    "charge overcurrent". What they agree on is the coarse fact, and a coarse
    fact that is true beats a specific one that might not be.

    The raw bytes ride along in raw_register so a bench session against a real
    pack can settle the layout without another firmware round trip.
    """
    if len(data) < _ALARM_BYTES:
        return []
    flagged = any(byte for byte in data[:_ALARM_BYTES])
    return [
        Reading(
            metric_name="battery.state",
            value=1.0 if flagged else 0.0,
            timestamp=now,
            quality=0,
            raw_register=(
                f"can:0x{ALARM_FRAME_ID:03X}:"
                + ",".join(f"{byte:02X}" for byte in data[:_ALARM_BYTES])
            ),
            asset_id=asset_id,
        )
    ]


def decode_identity(arb_id: int, data: bytes) -> tuple[str, str] | None:
    """Decode an identity frame into (field, text), or None.

    Identity is never telemetry. It goes to the adapter's identity block and
    from there into a validation record.
    """
    field = IDENTITY_FRAMES.get(arb_id)
    if field is None:
        return None
    text = bytes(data).decode("ascii", errors="replace").replace("\x00", "").strip()
    return (field, text) if text else None


def supported_metrics() -> list[str]:
    """Every metric the shared profile can produce."""
    metrics = [
        metric for frame in VICTRON_FRAMES.values() for metric in frame["fields"]
    ]
    metrics.append("battery.state")
    return sorted(set(metrics))
