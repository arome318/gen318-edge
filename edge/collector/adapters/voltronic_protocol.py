"""The Voltronic Power ASCII command protocol — framing, CRC and parsers.

Voltronic's inverter core is the single most common one in the Nigerian
market. It ships as Axpert, and then under a long list of other badges —
MPP Solar, Sako, EASUN, PowMr, Greco, Mecer and a dozen importer labels. A
site is far more likely to have one of these than to know it does.

IT IS NOT MODBUS. It is an ASCII command protocol with a binary checksum,
over 2400 baud serial or a USB HID interface, and none of the Modbus
machinery in this package applies to it. That is why it gets its own adapter
rather than a vendor map.

Two incompatible generations are in the field and a unit tells you which it
speaks only by answering. PI30 sends a bare command and gets back a
parenthesis-prefixed, space-delimited reply. PI18 wraps both directions in a
length-prefixed envelope and comes back comma-delimited. The adapter probes
PI30 first and falls back to PI18.

SOURCES (full ledger: docs/vendor-integrations/SOURCES.md)
  V1 = jblance/mpp-solar @ eafdd432 (MIT) — protocols/pi30.py, pi18.py,
       abstractprotocol.py, protocol_helpers.py, inout/hidrawio.py.
  V2 = "HS_MS_MSX RS232 Protocol", the vendor's own document dated
       2014-08-22, mirrored in ned-kelly/docker-voltronic-homeassistant.
  V3 = the C reference implementation in that same repo,
       sources/inverter-cli/inverter.cpp.
  Read 19 September 2026.

READ-ONLY BY CONSTRUCTION. Voltronic's setter commands are not marked by any
character in the command name — `PBT` sets battery type and `QPI` asks for
the protocol version, and nothing about the strings says which is which. So
this module refuses to frame anything that is not on an explicit allowlist,
rather than trying to spot a setter by its shape. ``write()`` on the adapter
always returns False.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ── CRC-16/XMODEM, as Voltronic computes it ─────────────────────────────
#
# Poly 0x1021, init 0x0000, no reflection, no final xor. V1 and V3 compute it
# a nibble at a time through this 16-entry table; V1's PI17 path uses an
# ordinary byte-wise implementation instead. All three agree with each other
# and with the published check value for "123456789" (0x31C3), which
# tests/test_voltronic_inverter.py asserts rather than taking on trust.
# fmt: off
_CRC_TABLE: tuple[int, ...] = (
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
    0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
)
# fmt: on

#: Checksum bytes that would collide with framing and are incremented by one
#: instead. 0x28 is the response's opening "(", 0x0D the terminator, 0x0A a
#: stray newline.
#:
#: V1 ALSO substitutes 0x00 here; V3, the C reference, does not. That
#: disagreement is real and unresolved, and this module takes the three bytes
#: both agree on. It cannot bite: no command on READ_COMMANDS produces a
#: checksum containing a 0x00 byte, which is asserted by a test so that
#: adding a command re-opens the question rather than silently inheriting a
#: choice made for a different command set.
_SUBSTITUTED = (0x28, 0x0D, 0x0A)

TERMINATOR = 0x0D


def crc16_xmodem(data: bytes) -> int:
    """The raw checksum, before Voltronic's byte substitution."""
    crc = 0
    for byte in data:
        high = ((crc >> 8) & 0xFF) >> 4
        crc = (crc << 4) & 0xFFFF
        crc ^= _CRC_TABLE[high ^ (byte >> 4)]
        high = ((crc >> 8) & 0xFF) >> 4
        crc = (crc << 4) & 0xFFFF
        crc ^= _CRC_TABLE[high ^ (byte & 0x0F)]
    return crc


def crc_bytes(data: bytes) -> tuple[int, int]:
    """The two checksum bytes as they go on the wire, high byte first."""
    crc = crc16_xmodem(data)
    high = (crc >> 8) & 0xFF
    low = crc & 0xFF
    if high in _SUBSTITUTED:
        high += 1
    if low in _SUBSTITUTED:
        low += 1
    return high, low


# ── The command allowlist ───────────────────────────────────────────────
#
# A Voltronic setter is not distinguishable from a query by inspection. V1
# lists 26 PI30 setters (F, PBT, PCP, POP, DAT, PE, PD and the rest) that
# change output frequency, charger priority, battery chemistry and the device
# clock, and they look exactly like queries on the wire. PI18 marks setters
# with a "^S" prefix, which is checkable — but PI30 offers nothing.
#
# So the rule here is an allowlist, not a denylist. Anything not named below
# is refused before it can be framed.

#: PI30: sent bare, answered with "(" + fields + checksum + CR.
PI30_COMMANDS: frozenset[str] = frozenset(
    {"QPI", "QID", "QSID", "QVFW", "QVFW2", "QVFW3", "QMOD", "QPIGS", "QPIRI", "QPIWS"}
)

#: PI18: the query envelope is "^P" + a three-digit length. The setter
#: envelope is "^S", and nothing here uses it.
PI18_COMMANDS: dict[str, str] = {
    "PI": "^P005",
    "ID": "^P005",
    "VFW": "^P006",
    "GS": "^P005",
    "MOD": "^P006",
    "PIRI": "^P007",
}

READ_COMMANDS: frozenset[str] = PI30_COMMANDS | frozenset(PI18_COMMANDS)


class VoltronicProtocolError(ValueError):
    """A frame could not be built or a reply could not be trusted."""


def frame_pi30(command: str) -> bytes:
    """``command + checksum + CR``. There is no leading "(" on a request."""
    if command not in PI30_COMMANDS:
        raise VoltronicProtocolError(
            f"{command!r} is not an allowed PI30 read command. Voltronic setters "
            f"look identical to queries on the wire, so only the allowlist may be sent."
        )
    payload = command.encode("ascii")
    high, low = crc_bytes(payload)
    return payload + bytes((high, low, TERMINATOR))


def frame_pi18(command: str) -> bytes:
    """``^Pnnn + command + checksum + CR``. The checksum covers the prefix."""
    prefix = PI18_COMMANDS.get(command)
    if prefix is None:
        raise VoltronicProtocolError(
            f"{command!r} is not an allowed PI18 read command. Only the '^P' query "
            f"envelope is ever framed here; '^S' is Voltronic's setter envelope."
        )
    payload = (prefix + command).encode("ascii")
    high, low = crc_bytes(payload)
    return payload + bytes((high, low, TERMINATOR))


# ── Reply framing ───────────────────────────────────────────────────────

#: PI30 rejects a command with a literal "(NAK". PI18 uses fixed 5-byte
#: frames, which are exactly crc_bytes(b"^0") and crc_bytes(b"^1").
PI18_NAK = b"^0\x1b\xe3\r"
PI18_ACK = b"^1\x0b\xc2\r"


@dataclass(frozen=True, slots=True)
class Reply:
    """A verified reply, split into its fields."""

    fields: tuple[str, ...]
    raw: bytes


def parse_pi30_reply(raw: bytes, *, verify_crc: bool = True) -> Reply:
    """Strip "(", verify the checksum, split on spaces.

    V3 checks three things before trusting a reply — the opening "(", the
    trailing CR, and the checksum — and so does this.
    """
    if len(raw) < 4:
        raise VoltronicProtocolError(f"reply too short to be a frame: {raw!r}")
    if raw[0:1] != b"(":
        raise VoltronicProtocolError(f"reply does not open with '(': {raw!r}")
    if raw[-1] != TERMINATOR:
        raise VoltronicProtocolError(f"reply does not end with CR: {raw!r}")
    body = raw[:-3]
    if verify_crc:
        expected = crc_bytes(body)
        actual = (raw[-3], raw[-2])
        if expected != actual:
            raise VoltronicProtocolError(
                f"checksum mismatch: computed {expected!r}, frame carries {actual!r}"
            )
    payload = body[1:].decode("ascii", errors="replace").strip()
    if payload == "NAK":
        raise VoltronicProtocolError("device returned NAK")
    return Reply(tuple(payload.split(" ")), raw)


def parse_pi18_reply(raw: bytes, *, verify_crc: bool = True) -> Reply:
    """Strip the ``^Dnnn`` envelope, verify the checksum, split on commas.

    V1 does NOT verify the checksum on PI18 replies — it overrides the check
    for PI30 and leaves PI18 on a base class that only tests for None. That
    is a gap in the reference implementation, not a property of the protocol,
    so the checksum is checked here in both directions.
    """
    if raw == PI18_NAK:
        raise VoltronicProtocolError("device returned NAK")
    if len(raw) < 8:
        raise VoltronicProtocolError(f"reply too short to be a frame: {raw!r}")
    if raw[0:2] != b"^D":
        raise VoltronicProtocolError(f"reply does not open with '^D': {raw!r}")
    if raw[-1] != TERMINATOR:
        raise VoltronicProtocolError(f"reply does not end with CR: {raw!r}")
    body = raw[:-3]
    if verify_crc:
        expected = crc_bytes(body)
        actual = (raw[-3], raw[-2])
        if expected != actual:
            raise VoltronicProtocolError(
                f"checksum mismatch: computed {expected!r}, frame carries {actual!r}"
            )
    payload = body[5:].decode("ascii", errors="replace").strip()
    return Reply(tuple(payload.split(",")), raw)


# ── Field parsers ───────────────────────────────────────────────────────
#
# Every parser below returns only the fields its sources AGREE on, and
# returns a dict so a short reply loses fields rather than raising.


def _number(fields: tuple[str, ...], index: int) -> float | None:
    if index >= len(fields):
        return None
    text = fields[index].strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    # "nan" and "inf" are valid float() input and neither is a reading. Sync
    # tombstones a non-finite value and the row is then pruned, so a garbled
    # field would cost the whole batch rather than one number. Unreachable
    # while the checksum is verified; reachable through verify_crc: false.
    return value if math.isfinite(value) else None


#: QPIGS positions that V1 and V2 agree on, field-for-field. Index 4 (output
#: apparent power, VA), 6 (load percent), 7 (bus voltage) and 14 (battery
#: voltage from the solar charger) are agreed too but have no canonical
#: metric, so they are not carried.
#:
#: Index 12 is deliberately absent. V1's base pi30 calls it "PV input current
#: for battery" in amps; V1's own pi30revo calls the SAME position "PV power"
#: in watts. A different physical quantity at one wire position is not
#: something a reading can be published through.
_QPIGS_FIELDS: tuple[tuple[int, str, float], ...] = (
    (0, "grid.voltage_v", 1.0),
    (1, "grid.frequency_hz", 1.0),
    (2, "inverter.ac_output_voltage_v", 1.0),
    (3, "inverter.ac_output_freq_hz", 1.0),
    (5, "inverter.ac_output_power_w", 1.0),
    (8, "battery.voltage_v", 1.0),
    (10, "battery.soc_pct", 1.0),
    (11, "inverter.temperature_c", 1.0),
    (13, "solar.pv_voltage_v", 1.0),
)

#: Charging and discharging current arrive as two separate unsigned
#: magnitudes, at index 9 and index 15. That makes the direction explicit on
#: the wire rather than a convention to be guessed, which is the opposite of
#: every Modbus map in this package — and the reason battery current is
#: published here and omitted from several of them.
_QPIGS_CHARGE_A = 9
_QPIGS_DISCHARGE_A = 15

#: Index 19 is PV input power in watts. It is in V1 and NOT in V2, whose
#: revision stops at index 16 — a later firmware extension the 2014 document
#: predates. Its presence is self-describing: a reply carrying it is longer.
#: Read only when the reply is actually long enough.
_QPIGS_PV_POWER = 19


def parse_qpigs(fields: tuple[str, ...]) -> dict[str, float]:
    """General status. Returns canonical metric names to values."""
    out: dict[str, float] = {}
    for index, metric, scale in _QPIGS_FIELDS:
        value = _number(fields, index)
        if value is not None:
            out[metric] = value * scale

    charge = _number(fields, _QPIGS_CHARGE_A)
    discharge = _number(fields, _QPIGS_DISCHARGE_A)
    if charge is not None and discharge is not None:
        # Positive is charge, which is the platform's convention.
        out["battery.current_a"] = charge - discharge

    pv_power = _number(fields, _QPIGS_PV_POWER)
    if pv_power is not None:
        out["solar.pv_power_w"] = pv_power
    return out


#: QMOD returns a letter. P/S/L/B/F/H are in all three sources; Y is in V1
#: only and absent from both the vendor document and the C reference, so it
#: is a later addition. The numbers are this adapter's own stable codes —
#: inverter.state carries vendor-local codes everywhere in this package, so
#: they are not comparable across brands and are documented, not derived.
QMOD_STATES: dict[str, int] = {
    "P": 0,  # power on
    "S": 1,  # standby
    "L": 2,  # line / running from the grid
    "B": 3,  # battery
    "F": 4,  # fault
    "H": 5,  # power saving
    "Y": 6,  # bypass — V1 only, not in the vendor document
}


def parse_qmod(fields: tuple[str, ...]) -> dict[str, float]:
    if not fields:
        return {}
    code = fields[0].strip()[:1]
    state = QMOD_STATES.get(code)
    if state is None:
        # An unmapped letter is exactly where a guess is dangerous: a new
        # firmware's mode would be indistinguishable from a fault.
        return {}
    return {"inverter.state": float(state)}


def parse_qpiws(fields: tuple[str, ...]) -> dict[str, float]:
    """Warning and fault flags, as the 1-based position of the first set bit.

    V1 and V2 agree on all 32 bit names and their order. They disagree only
    on whether a17 and a27 are labelled fault or warning, which changes no
    number. V1's own two captured fixtures are 34 and 38 characters long
    against a 32-bit table, so the length is not reliable and only the
    positions that exist are read.
    """
    if not fields:
        return {}
    flags = fields[0].strip()
    for position, char in enumerate(flags[:32], start=1):
        if char == "1":
            return {"inverter.error_code": float(position)}
    return {"inverter.error_code": 0.0}


#: PI18's GS carries both trackers and the mode's own temperatures in one
#: reply, where PI30 needs a second command. Tenths are integers here, not
#: decimal strings, so each field carries its own scale.
_PI18_GS_FIELDS: tuple[tuple[int, str, float], ...] = (
    (0, "grid.voltage_v", 0.1),
    (1, "grid.frequency_hz", 0.1),
    (2, "inverter.ac_output_voltage_v", 0.1),
    (3, "inverter.ac_output_freq_hz", 0.1),
    (5, "inverter.ac_output_power_w", 1.0),
    (7, "battery.voltage_v", 0.1),
    (12, "battery.soc_pct", 1.0),
    (13, "inverter.temperature_c", 1.0),
    (16, "solar.mppt1_power_w", 1.0),
    (17, "solar.mppt2_power_w", 1.0),
    (18, "solar.mppt1_voltage_v", 0.1),
    (19, "solar.mppt2_voltage_v", 0.1),
)

_PI18_GS_DISCHARGE_A = 10
_PI18_GS_CHARGE_A = 11


def parse_pi18_gs(fields: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for index, metric, scale in _PI18_GS_FIELDS:
        value = _number(fields, index)
        if value is not None:
            out[metric] = value * scale

    charge = _number(fields, _PI18_GS_CHARGE_A)
    discharge = _number(fields, _PI18_GS_DISCHARGE_A)
    if charge is not None and discharge is not None:
        out["battery.current_a"] = charge - discharge

    mppt1 = out.get("solar.mppt1_power_w")
    mppt2 = out.get("solar.mppt2_power_w")
    if mppt1 is not None and mppt2 is not None:
        out["solar.pv_power_w"] = mppt1 + mppt2
    return out


#: PI18's MOD is a numeric index, not a letter — a different encoding, not a
#: renamed field. Mapped onto the same adapter-local state codes as QMOD so
#: one site does not change meaning when its inverter is replaced.
PI18_MODES: dict[int, int] = {
    0: 0,  # power on
    1: 1,  # standby
    2: 6,  # bypass
    3: 3,  # battery
    4: 4,  # fault
    5: 2,  # hybrid (line / grid) — PI30's "line"
}


def parse_pi18_mod(fields: tuple[str, ...]) -> dict[str, float]:
    if not fields:
        return {}
    try:
        code = int(fields[0].strip())
    except ValueError:
        return {}
    state = PI18_MODES.get(code)
    if state is None:
        return {}
    return {"inverter.state": float(state)}


#: Every metric either generation can produce. The adapter reports this as
#: its capability set regardless of which one a given unit turns out to
#: speak, because that is not known until it answers.
SUPPORTED_METRICS: tuple[str, ...] = (
    "grid.voltage_v",
    "grid.frequency_hz",
    "inverter.ac_output_voltage_v",
    "inverter.ac_output_freq_hz",
    "inverter.ac_output_power_w",
    "inverter.temperature_c",
    "inverter.state",
    "inverter.error_code",
    "battery.voltage_v",
    "battery.current_a",
    "battery.soc_pct",
    "solar.pv_voltage_v",
    "solar.pv_power_w",
    "solar.mppt1_voltage_v",
    "solar.mppt1_power_w",
    "solar.mppt2_voltage_v",
    "solar.mppt2_power_w",
)
