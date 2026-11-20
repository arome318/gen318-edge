"""Decoding Modbus words into readings, for vendor maps and the probe.

`generic_modbus` historically understood five types, all big-endian, one
register per request. Real vendor maps need more: Deye's 32-bit counters put
the LOW word first, DSE puts the HIGH word first, temperatures arrive as
``raw * 0.1 - 100`` on some firmware, run state is an enum, breaker position
is one bit inside a word, and a fault is four words of bit flags.

The one rule running through every decoder here: **absence is reported, never
invented.** A sentinel the map declares, an enum code the map does not list, a
short response, a non-finite float and a u64 too large for the ``Reading``
contract all return a `Decoded` with ``value=None`` and a named ``refusal``.
None of them return 0.0 — a fabricated zero on a power register reads as a
healthy site with nothing running, and nothing downstream can tell it from a
measurement.

Bounds work the other way. A value outside the map's declared ``min``/``max``
is still data: it is emitted with ``quality=1`` (suspect) and the catalog's own
envelope remains the second gate at ingest. Never dropped.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

from .vendor_map import RegisterSpec

# A Reading carries a float, and IEEE-754 doubles hold integers exactly only up
# to 2**53. A 64-bit counter above that would be silently rounded — for a
# lifetime energy counter that is a wrong number presented as a measurement, so
# the decoder refuses instead. Documented in the vendor_maps README.
MAX_EXACT_INTEGER = 2**53

QUALITY_GOOD = 0
QUALITY_SUSPECT = 1

# Refusal reasons. These are evidence: they reach the probe's validation record
# and the debug log, so a technician can tell "the device said nothing" from
# "the map is wrong".
REFUSAL_SHORT_READ = "short_read"
REFUSAL_NA_SENTINEL = "na_sentinel"
REFUSAL_UNKNOWN_ENUM = "unknown_enum"
REFUSAL_NON_FINITE = "non_finite"
REFUSAL_U64_PRECISION = "u64_precision"
REFUSAL_MISSING_COMPONENT = "missing_component"


@dataclass(frozen=True, slots=True)
class Decoded:
    """The outcome of decoding one register's words.

    ``value is None`` always means "no reading" and ``refusal`` says why.
    ``text`` carries identity strings, which are never telemetry.
    """

    value: float | None = None
    quality: int = QUALITY_GOOD
    text: str | None = None
    raw_words: tuple[int, ...] = field(default_factory=tuple)
    refusal: str | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None or self.text is not None


def words_needed(spec: RegisterSpec) -> int:
    """Registers this spec reads. Derived entries read nothing themselves."""
    return 0 if spec.derived else spec.word_count


# ── Primitive word assembly ─────────────────────────────────────


def _u16(words: tuple[int, ...]) -> int:
    return words[0] & 0xFFFF


def _be(words: tuple[int, ...]) -> int:
    """Big-endian word order: the LOWER address holds the MOST significant word.

    Victron and DSE write 32-bit values this way.
    """
    value = 0
    for word in words:
        value = (value << 16) | (word & 0xFFFF)
    return value


def _le(words: tuple[int, ...]) -> int:
    """Word-swapped order: the LOWER address holds the LEAST significant word.

    Deye's U32 counters (516/517, 534/535 …) are documented this way by every
    community source; getting it backwards turns 12.3 kWh into 806,092 kWh.
    """
    value = 0
    for word in reversed(words):
        value = (value << 16) | (word & 0xFFFF)
    return value


def _signed(value: int, bits: int) -> int:
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def _raw_integer(spec: RegisterSpec, words: tuple[int, ...]) -> int:
    """The register's unscaled integer value, before sign, scale and offset."""
    data_type = spec.data_type
    if spec.high_word_offset is not None:
        # Low word first, high word ``high_word_offset`` registers later; the
        # words in between belong to other registers and are not part of it.
        words = (words[0], words[spec.high_word_offset])
    if data_type in ("uint16", "enum", "bitfield", "fault_bits"):
        return _u16(words) if len(words) == 1 else _be(words)
    if data_type == "int16":
        return _signed(_u16(words), 16)
    if data_type == "uint32":
        return _be(words)
    if data_type == "int32":
        return _signed(_be(words), 32)
    if data_type == "uint32_le":
        return _le(words)
    if data_type == "int32_le":
        return _signed(_le(words), 32)
    if data_type == "uint64":
        return _be(words)
    if data_type == "uint64_le":
        return _le(words)
    raise ValueError(f"not an integer decoder: {data_type}")


# ── The public decoder ──────────────────────────────────────────


def decode_register(spec: RegisterSpec, words: list[int] | tuple[int, ...]) -> Decoded:
    """Decode one register's words into a reading, or a named refusal."""
    raw = tuple(int(word) & 0xFFFF for word in words)
    if len(raw) < spec.word_count:
        return Decoded(raw_words=raw, refusal=REFUSAL_SHORT_READ)
    raw = raw[: spec.word_count]

    if spec.data_type == "string":
        return _decode_string(raw)

    if spec.na_values:
        # A sentinel is only a sentinel when the MAP says so. Assuming 0xFFFF
        # means "not implemented" would discard a real 65535 on a vendor that
        # has no such convention, which is most of them.
        unscaled = _raw_unsigned_for_sentinel(spec, raw)
        if unscaled in spec.na_values:
            return Decoded(raw_words=raw, refusal=REFUSAL_NA_SENTINEL)

    if spec.data_type == "enum":
        return _decode_enum(spec, raw)
    if spec.data_type == "bitfield":
        return _decode_bitfield(spec, raw)
    if spec.data_type == "fault_bits":
        return _decode_fault_bits(spec, raw)
    if spec.data_type in ("float32", "float32_le"):
        return _finish(spec, _decode_float(spec, raw), raw)

    integer = _raw_integer(spec, raw)
    if spec.data_type.startswith("uint64") and integer > MAX_EXACT_INTEGER:
        return Decoded(raw_words=raw, refusal=REFUSAL_U64_PRECISION)
    return _finish(spec, float(integer), raw)


def _raw_unsigned_for_sentinel(spec: RegisterSpec, raw: tuple[int, ...]) -> int:
    """The bit pattern a declared ``na:`` value is compared against.

    Sentinels are written as the vendor prints them — 65535, 0xFFFFFFFF — so
    the comparison is on the UNSIGNED pattern in the map's word order, before
    any sign interpretation.
    """
    if spec.data_type in ("uint32_le", "int32_le", "uint64_le", "float32_le"):
        return _le(raw)
    if len(raw) == 1:
        return _u16(raw)
    return _be(raw)


def _decode_string(raw: tuple[int, ...]) -> Decoded:
    """Two ASCII bytes per register, high byte first; trailing NULs trimmed."""
    data = b"".join(struct.pack(">H", word) for word in raw)
    text = data.decode("ascii", errors="replace").replace("\x00", "").strip()
    return Decoded(text=text or None, raw_words=raw)


def _decode_enum(spec: RegisterSpec, raw: tuple[int, ...]) -> Decoded:
    table = spec.enum_values or {}
    code = _u16(raw)
    if code not in table:
        # An unmapped state code is exactly the case where a guess is
        # dangerous: emitting the raw code as if it were a mapped state makes
        # a new firmware's "grid-forming" indistinguishable from "fault".
        return Decoded(raw_words=raw, refusal=REFUSAL_UNKNOWN_ENUM)
    return _finish(spec, table[code], raw, apply_scale=False)


def _decode_bitfield(spec: RegisterSpec, raw: tuple[int, ...]) -> Decoded:
    value = _be(raw) if len(raw) > 1 else _u16(raw)
    mask = spec.mask or 0
    return _finish(spec, 1.0 if value & mask else 0.0, raw, apply_scale=False)


def _decode_fault_bits(spec: RegisterSpec, raw: tuple[int, ...]) -> Decoded:
    """First set fault bit, as a 1-based position across the whole block.

    Deye spreads its faults over four words. 64 bits do not fit a float
    exactly, so the raw pattern cannot be the reading; the position of the
    first set bit can, and the raw words ride along as evidence for the
    fault-code table a bench session will build.

    0.0 means no bit is set anywhere, which is a real observation (no fault),
    not an absence. Positions are 1-based precisely so that 0 stays
    unambiguous: bit 0 of word 0 decodes to 1.0.
    """
    for word_index, word in enumerate(raw):
        if not word:
            continue
        for bit in range(16):
            if word & (1 << bit):
                return _finish(
                    spec, float(word_index * 16 + bit + 1), raw, apply_scale=False
                )
    return _finish(spec, 0.0, raw, apply_scale=False)


def _decode_float(spec: RegisterSpec, raw: tuple[int, ...]) -> float:
    ordered = raw if spec.data_type == "float32" else tuple(reversed(raw))
    return float(struct.unpack(">f", struct.pack(">HH", ordered[0], ordered[1]))[0])


def _finish(
    spec: RegisterSpec,
    value: float,
    raw: tuple[int, ...],
    *,
    apply_scale: bool = True,
) -> Decoded:
    """Apply scale then offset, check declared bounds, reject non-finite.

    Scale before offset is the documented order and it matters: Deye's
    temperatures are ``raw * 0.1 - 100``, which the other order turns into
    a reading around -1000 °C.
    """
    if apply_scale:
        value = value * spec.scale + spec.offset
    if not math.isfinite(value):
        return Decoded(raw_words=raw, refusal=REFUSAL_NON_FINITE)
    quality = QUALITY_GOOD
    if spec.min_value is not None and value < spec.min_value:
        quality = QUALITY_SUSPECT
    if spec.max_value is not None and value > spec.max_value:
        quality = QUALITY_SUSPECT
    return Decoded(value=value, quality=quality, raw_words=raw)


# ── Derived values ──────────────────────────────────────────────


def derive_sum(spec: RegisterSpec, decoded: dict[str, Decoded]) -> Decoded:
    """Sum the components of a derived register, or refuse.

    A partial sum is a wrong number that looks right: three of four MPPT
    strings summed still plots as plant power. So the sum exists only when
    EVERY component produced a value this poll; one missing component refuses
    the whole thing. Quality is the worst of the components — a suspect input
    cannot produce a trusted total.
    """
    total = 0.0
    quality = QUALITY_GOOD
    for component in spec.components:
        part = decoded.get(component)
        if part is None or part.value is None:
            return Decoded(refusal=REFUSAL_MISSING_COMPONENT)
        total += part.value
        quality = max(quality, part.quality)
    value = total * spec.scale + spec.offset
    if not math.isfinite(value):
        return Decoded(refusal=REFUSAL_NON_FINITE)
    if spec.min_value is not None and value < spec.min_value:
        quality = max(quality, QUALITY_SUSPECT)
    if spec.max_value is not None and value > spec.max_value:
        quality = max(quality, QUALITY_SUSPECT)
    return Decoded(value=value, quality=quality)


def split_signed(spec: RegisterSpec, decoded: Decoded) -> tuple[str, float] | None:
    """The second half of a signed import/export register.

    One signed CT reading carries both directions. The register's own
    ``metric`` takes the positive side and ``split_negative_to`` the magnitude
    of the negative side; the other is emitted as a real 0.0, because "not
    exporting" is a measurement, not a gap.

    Which sign means which direction is a per-unit fact the map declares and a
    bench session confirms — every map shipping this carries the uncertainty
    marker on that line.
    """
    if spec.split_negative_to is None or decoded.value is None:
        return None
    value = decoded.value
    return (spec.split_negative_to, abs(value) if value < 0 else 0.0)


def primary_value(spec: RegisterSpec, decoded: Decoded) -> float | None:
    """The value for the spec's own metric, honouring a signed split."""
    if decoded.value is None:
        return None
    if spec.split_negative_to is None:
        return decoded.value
    return decoded.value if decoded.value > 0 else 0.0
