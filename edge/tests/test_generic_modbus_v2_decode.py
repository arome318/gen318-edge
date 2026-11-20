"""Modbus decoding vectors (vendor integrations spec §2.1, plan Task 3).

Every decoder is pinned against a FIXED word sequence written out by hand
here, independent of the implementation — a test that builds its input with
the same helper the code uses proves only that the helper is self-consistent.

The theme these tests protect is the one the spec keeps returning to:
absence is reported, never invented. A declared sentinel, an unmapped enum
code, a short response, a non-finite float and a u64 too big for a float all
produce NO reading and a named reason. None of them produce 0.0.
"""

from __future__ import annotations

import struct
from typing import Any

import pytest

from collector.adapters import modbus_decode as md
from collector.adapters.modbus_decode import Decoded, decode_register, derive_sum
from collector.adapters.vendor_map import RegisterSpec


def _spec(**overrides: Any) -> RegisterSpec:
    base: dict[str, Any] = {
        "id": "r",
        "metric": "battery.voltage_v",
        "address": 100,
        "data_type": "uint16",
        "word_count": 1,
        "asset_role": "battery",
        "group": "power",
        "function_code": 3,
        "unit_id": 1,
    }
    base.update(overrides)
    return RegisterSpec(**base)


# ── Unsigned and signed integers, including extrema ─────────────


def test_uint16_vector() -> None:
    assert decode_register(_spec(), [0x1234]).value == 4660.0


@pytest.mark.parametrize(
    "word, expected",
    [
        (0x0000, 0.0),
        (0x0001, 1.0),
        (0x7FFF, 32767.0),  # positive extremum
        (0x8000, -32768.0),  # negative extremum
        (0xFFFF, -1.0),
    ],
)
def test_int16_sign_extrema(word: int, expected: float) -> None:
    assert decode_register(_spec(data_type="int16"), [word]).value == expected


def test_uint16_does_not_sign_extend() -> None:
    assert decode_register(_spec(), [0xFFFF]).value == 65535.0


# ── 32-bit word order: the defect that turns 12 kWh into 800,000 ──


def test_uint32_big_endian_puts_the_high_word_first() -> None:
    spec = _spec(data_type="uint32", word_count=2)
    assert decode_register(spec, [0x0001, 0x0002]).value == float((1 << 16) | 2)


def test_uint32_le_puts_the_low_word_first() -> None:
    """Deye's counters: the LOWER address is the LEAST significant word."""
    spec = _spec(data_type="uint32_le", word_count=2)
    assert decode_register(spec, [0x0002, 0x0001]).value == float((1 << 16) | 2)


def test_the_two_word_orders_disagree_on_the_same_words() -> None:
    words = [0x0001, 0x0002]
    be = decode_register(_spec(data_type="uint32", word_count=2), words).value
    le = decode_register(_spec(data_type="uint32_le", word_count=2), words).value
    assert be == 65538.0
    assert le == 131073.0
    assert be != le


@pytest.mark.parametrize(
    "data_type, words, expected",
    [
        ("int32", [0xFFFF, 0xFFFF], -1.0),
        ("int32", [0x8000, 0x0000], -2147483648.0),
        ("int32", [0x7FFF, 0xFFFF], 2147483647.0),
        ("int32_le", [0xFFFF, 0xFFFF], -1.0),
        ("int32_le", [0x0000, 0x8000], -2147483648.0),
        ("int32_le", [0xFFFF, 0x7FFF], 2147483647.0),
    ],
)
def test_int32_sign_extrema_both_word_orders(
    data_type: str, words: list[int], expected: float
) -> None:
    spec = _spec(data_type=data_type, word_count=2)
    assert decode_register(spec, words).value == expected


# ── A 32-bit value whose high word is not adjacent ──────────────


def test_split_uint32_le_skips_the_word_between_its_halves() -> None:
    """Deye single-phase grid import: low at 78, frequency at 79, high at 80.

    5002 (50.02 Hz) in the middle word must play no part in the value.
    """
    spec = _spec(data_type="uint32_le", word_count=3, high_word_offset=2)
    decoded = decode_register(spec, [0x86A0, 5002, 0x0001])  # 100000
    assert decoded.value == 100000.0
    assert decoded.raw_words == (0x86A0, 5002, 0x0001)


def test_reading_the_split_counter_as_adjacent_words_is_wildly_wrong() -> None:
    """The defect the offset exists to prevent: frequency in the high word."""
    adjacent = _spec(data_type="uint32_le", word_count=2)
    wrong = decode_register(adjacent, [0x86A0, 5002]).value
    assert wrong == float(5002 * 65536 + 0x86A0)
    assert wrong != 100000.0


def test_split_int32_le_keeps_its_sign() -> None:
    spec = _spec(data_type="int32_le", word_count=3, high_word_offset=2)
    assert decode_register(spec, [0xFFFF, 0x1234, 0xFFFF]).value == -1.0


def test_split_register_short_read_is_refused_not_zeroed() -> None:
    spec = _spec(data_type="uint32_le", word_count=3, high_word_offset=2)
    decoded = decode_register(spec, [0x0001, 0x0002])
    assert decoded.value is None
    assert decoded.refusal == md.REFUSAL_SHORT_READ


# ── u64 and the float Reading contract ──────────────────────────


def test_uint64_decodes_below_the_exact_integer_limit() -> None:
    spec = _spec(data_type="uint64", word_count=4)
    assert decode_register(spec, [0x0000, 0x0000, 0x0000, 0x0005]).value == 5.0


def test_uint64_le_reverses_the_word_order() -> None:
    spec = _spec(data_type="uint64_le", word_count=4)
    assert decode_register(spec, [0x0005, 0x0000, 0x0000, 0x0000]).value == 5.0


def test_uint64_at_the_exact_limit_is_still_decoded() -> None:
    """2**53 is the largest integer a float holds exactly; it is allowed."""
    spec = _spec(data_type="uint64", word_count=4)
    words = [0x0020, 0x0000, 0x0000, 0x0000]  # 2**53
    assert md.MAX_EXACT_INTEGER == 2**53
    assert decode_register(spec, words).value == float(2**53)


def test_uint64_beyond_float_precision_refuses_rather_than_rounding() -> None:
    spec = _spec(data_type="uint64", word_count=4)
    words = [0x0020, 0x0000, 0x0000, 0x0001]  # 2**53 + 1, not representable
    result = decode_register(spec, words)
    assert result.value is None
    assert result.refusal == md.REFUSAL_U64_PRECISION
    assert result.raw_words == (0x0020, 0x0000, 0x0000, 0x0001)


# ── float32 ─────────────────────────────────────────────────────


def test_float32_big_endian() -> None:
    high, low = struct.unpack(">HH", struct.pack(">f", 3.25))
    spec = _spec(data_type="float32", word_count=2)
    assert decode_register(spec, [high, low]).value == pytest.approx(3.25)


def test_float32_le_swaps_the_words() -> None:
    high, low = struct.unpack(">HH", struct.pack(">f", 3.25))
    spec = _spec(data_type="float32_le", word_count=2)
    assert decode_register(spec, [low, high]).value == pytest.approx(3.25)


def test_float32_nan_is_refused_not_emitted() -> None:
    spec = _spec(data_type="float32", word_count=2)
    result = decode_register(spec, [0x7FC0, 0x0000])  # quiet NaN
    assert result.value is None
    assert result.refusal == md.REFUSAL_NON_FINITE


def test_float32_infinity_is_refused() -> None:
    spec = _spec(data_type="float32", word_count=2)
    result = decode_register(spec, [0x7F80, 0x0000])
    assert result.value is None
    assert result.refusal == md.REFUSAL_NON_FINITE


# ── Scale then offset ───────────────────────────────────────────


def test_scale_is_applied_before_offset() -> None:
    """Deye temperature: raw * 0.1 - 100. The other order gives ~-1000 degC."""
    spec = _spec(metric="inverter.temperature_c", scale=0.1, offset=-100.0)
    assert decode_register(spec, [1250]).value == pytest.approx(25.0)


def test_scale_without_offset_is_unchanged() -> None:
    assert decode_register(_spec(scale=0.01), [5432]).value == pytest.approx(54.32)


def test_negative_scale_and_offset_compose() -> None:
    spec = _spec(data_type="int16", scale=0.1, offset=5.0)
    assert decode_register(spec, [0xFFF6]).value == pytest.approx(4.0)  # -10*0.1+5


# ── Declared sentinels ──────────────────────────────────────────


def test_declared_sentinel_refuses() -> None:
    spec = _spec(na_values=(65535,))
    result = decode_register(spec, [65535])
    assert result.value is None
    assert result.refusal == md.REFUSAL_NA_SENTINEL


def test_an_undeclared_sentinel_value_is_a_real_reading() -> None:
    """0xFFFF only means 'not implemented' where the vendor says it does."""
    result = decode_register(_spec(), [65535])
    assert result.value == 65535.0
    assert result.refusal is None


def test_sentinel_compares_the_pattern_in_the_maps_word_order() -> None:
    spec = _spec(data_type="uint32_le", word_count=2, na_values=(0xFFFFFFFF,))
    assert decode_register(spec, [0xFFFF, 0xFFFF]).refusal == md.REFUSAL_NA_SENTINEL
    assert decode_register(spec, [0x0001, 0x0000]).value == 1.0


def test_sentinel_is_checked_before_scaling() -> None:
    spec = _spec(scale=0.1, offset=-100.0, na_values=(65535,))
    assert decode_register(spec, [65535]).refusal == md.REFUSAL_NA_SENTINEL


# ── Enums ───────────────────────────────────────────────────────


def test_enum_maps_raw_codes_through_the_table() -> None:
    spec = _spec(
        metric="inverter.state", data_type="enum", enum_values={0: 0.0, 2: 2.0, 3: 3.0}
    )
    assert decode_register(spec, [2]).value == 2.0
    assert decode_register(spec, [0]).value == 0.0


def test_unknown_enum_code_refuses_rather_than_passing_the_raw_through() -> None:
    spec = _spec(
        metric="inverter.state", data_type="enum", enum_values={0: 0.0, 2: 2.0}
    )
    result = decode_register(spec, [9])
    assert result.value is None
    assert result.refusal == md.REFUSAL_UNKNOWN_ENUM
    assert result.raw_words == (9,)


def test_enum_ignores_scale_and_offset() -> None:
    """A state code is a label; arithmetic on it is meaningless."""
    spec = _spec(
        metric="inverter.state",
        data_type="enum",
        enum_values={2: 2.0},
        scale=0.1,
        offset=-100.0,
    )
    assert decode_register(spec, [2]).value == 2.0


# ── Bitfields ───────────────────────────────────────────────────


def test_bitfield_emits_one_when_any_masked_bit_is_set() -> None:
    spec = _spec(metric="genset.breaker_state", data_type="bitfield", mask=0b0100)
    assert decode_register(spec, [0b0100]).value == 1.0
    assert decode_register(spec, [0b1101]).value == 1.0


def test_bitfield_emits_zero_when_masked_bits_are_clear() -> None:
    spec = _spec(metric="genset.breaker_state", data_type="bitfield", mask=0b0100)
    assert decode_register(spec, [0b1011]).value == 0.0


def test_bitfield_spanning_two_words_is_big_endian() -> None:
    spec = _spec(
        metric="genset.breaker_state", data_type="bitfield", word_count=2, mask=1 << 16
    )
    assert decode_register(spec, [0x0001, 0x0000]).value == 1.0
    assert decode_register(spec, [0x0000, 0x0001]).value == 0.0


# ── Fault bits ──────────────────────────────────────────────────


def test_fault_bits_reports_the_first_set_bit_one_based() -> None:
    spec = _spec(metric="inverter.error_code", data_type="fault_bits", word_count=4)
    # bit 0 of word 0 -> position 1, so 0 stays unambiguously "no fault"
    assert decode_register(spec, [0x0001, 0, 0, 0]).value == 1.0
    # bit 3 of word 0 -> 4
    assert decode_register(spec, [0x0008, 0, 0, 0]).value == 4.0
    # bit 0 of word 1 -> 17
    assert decode_register(spec, [0x0000, 0x0001, 0, 0]).value == 17.0
    # bit 15 of word 3 -> 64
    assert decode_register(spec, [0, 0, 0, 0x8000]).value == 64.0


def test_fault_bits_returns_the_lowest_set_bit_when_several_are_set() -> None:
    spec = _spec(metric="inverter.error_code", data_type="fault_bits", word_count=4)
    assert decode_register(spec, [0x0000, 0x0006, 0x0001, 0]).value == 18.0


def test_fault_bits_zero_means_no_fault_and_is_a_real_reading() -> None:
    spec = _spec(metric="inverter.error_code", data_type="fault_bits", word_count=4)
    result = decode_register(spec, [0, 0, 0, 0])
    assert result.value == 0.0
    assert result.refusal is None


def test_fault_bits_carry_the_raw_words_as_evidence() -> None:
    spec = _spec(metric="inverter.error_code", data_type="fault_bits", word_count=4)
    result = decode_register(spec, [0x0000, 0x0006, 0x0001, 0x0000])
    assert result.raw_words == (0x0000, 0x0006, 0x0001, 0x0000)


# ── Bounds mark suspect, never drop ─────────────────────────────


def test_value_outside_declared_bounds_is_suspect_not_dropped() -> None:
    spec = _spec(min_value=0.0, max_value=100.0)
    result = decode_register(spec, [150])
    assert result.value == 150.0
    assert result.quality == md.QUALITY_SUSPECT


def test_value_below_the_declared_minimum_is_suspect() -> None:
    spec = _spec(data_type="int16", min_value=0.0, max_value=100.0)
    result = decode_register(spec, [0xFFFF])
    assert result.value == -1.0
    assert result.quality == md.QUALITY_SUSPECT


def test_value_inside_bounds_is_good_quality() -> None:
    spec = _spec(min_value=0.0, max_value=100.0)
    assert decode_register(spec, [50]).quality == md.QUALITY_GOOD


def test_bounds_are_checked_after_scaling() -> None:
    spec = _spec(scale=0.01, min_value=0.0, max_value=100.0)
    assert decode_register(spec, [5000]).quality == md.QUALITY_GOOD  # 50.0


# ── Short reads ─────────────────────────────────────────────────


def test_short_response_refuses_rather_than_decoding_partial_words() -> None:
    spec = _spec(data_type="uint32", word_count=2)
    result = decode_register(spec, [0x0001])
    assert result.value is None
    assert result.refusal == md.REFUSAL_SHORT_READ


def test_empty_response_refuses() -> None:
    assert decode_register(_spec(), []).refusal == md.REFUSAL_SHORT_READ


def test_extra_words_are_ignored_not_misread() -> None:
    spec = _spec(data_type="uint32", word_count=2)
    result = decode_register(spec, [0x0001, 0x0002, 0xDEAD])
    assert result.value == float((1 << 16) | 2)
    assert result.raw_words == (0x0001, 0x0002)


# ── Identity strings are never telemetry ────────────────────────


def test_string_decodes_ascii_and_carries_no_numeric_value() -> None:
    spec = _spec(metric="identity.serial", data_type="string", word_count=3)
    # "2305" + NUL padding, two bytes per register, high byte first
    words = [0x3233, 0x3035, 0x0000]
    result = decode_register(spec, words)
    assert result.text == "2305"
    assert result.value is None


def test_string_spec_is_marked_identity_only() -> None:
    spec = _spec(metric="identity.serial", data_type="string", word_count=3)
    assert spec.identity_only is True
    assert _spec().identity_only is False


def test_words_needed_matches_the_spec_width() -> None:
    assert md.words_needed(_spec()) == 1
    assert md.words_needed(_spec(data_type="uint32", word_count=2)) == 2
    # A derived register reads nothing of its own.
    assert md.words_needed(_spec(derived="sum", components=("a", "b"), address=-1)) == 0


# ── Derived sums ────────────────────────────────────────────────


def _ok(value: float, quality: int = md.QUALITY_GOOD) -> Decoded:
    return Decoded(value=value, quality=quality)


def test_derived_sum_adds_every_component() -> None:
    spec = _spec(
        metric="solar.pv_power_w",
        derived="sum",
        components=("pv1", "pv2"),
        address=-1,
        asset_role="solar_array",
    )
    result = derive_sum(spec, {"pv1": _ok(1200.0), "pv2": _ok(800.0)})
    assert result.value == 2000.0


def test_derived_sum_refuses_when_a_component_is_missing() -> None:
    """A partial sum plots as plant power and nothing downstream can tell."""
    spec = _spec(
        metric="solar.pv_power_w",
        derived="sum",
        components=("pv1", "pv2", "pv3"),
        address=-1,
        asset_role="solar_array",
    )
    result = derive_sum(spec, {"pv1": _ok(1200.0), "pv2": _ok(800.0)})
    assert result.value is None
    assert result.refusal == md.REFUSAL_MISSING_COMPONENT


def test_derived_sum_refuses_when_a_component_itself_refused() -> None:
    spec = _spec(
        metric="solar.pv_power_w",
        derived="sum",
        components=("pv1", "pv2"),
        address=-1,
        asset_role="solar_array",
    )
    refused = Decoded(refusal=md.REFUSAL_NA_SENTINEL)
    result = derive_sum(spec, {"pv1": _ok(1200.0), "pv2": refused})
    assert result.value is None
    assert result.refusal == md.REFUSAL_MISSING_COMPONENT


def test_derived_sum_inherits_the_worst_component_quality() -> None:
    spec = _spec(
        metric="solar.pv_power_w",
        derived="sum",
        components=("pv1", "pv2"),
        address=-1,
        asset_role="solar_array",
    )
    result = derive_sum(
        spec, {"pv1": _ok(1200.0), "pv2": _ok(800.0, md.QUALITY_SUSPECT)}
    )
    assert result.value == 2000.0
    assert result.quality == md.QUALITY_SUSPECT


def test_derived_sum_applies_its_own_scale() -> None:
    spec = _spec(
        metric="solar.pv_power_w",
        derived="sum",
        components=("pv1", "pv2"),
        address=-1,
        asset_role="solar_array",
        scale=10.0,
    )
    result = derive_sum(spec, {"pv1": _ok(100.0), "pv2": _ok(50.0)})
    assert result.value == 1500.0


# ── Signed import/export split ──────────────────────────────────


def test_positive_reading_goes_to_the_primary_metric_and_zero_to_the_other() -> None:
    spec = _spec(
        metric="grid.import_power_w",
        data_type="int16",
        split_negative_to="grid.export_power_w",
        asset_role="grid_connection",
    )
    decoded = decode_register(spec, [500])
    assert md.primary_value(spec, decoded) == 500.0
    assert md.split_signed(spec, decoded) == ("grid.export_power_w", 0.0)


def test_negative_reading_goes_to_the_split_metric_as_a_magnitude() -> None:
    spec = _spec(
        metric="grid.import_power_w",
        data_type="int16",
        split_negative_to="grid.export_power_w",
        asset_role="grid_connection",
    )
    decoded = decode_register(spec, [0xFE0C])  # -500
    assert decoded.value == -500.0
    assert md.primary_value(spec, decoded) == 0.0
    assert md.split_signed(spec, decoded) == ("grid.export_power_w", 500.0)


def test_both_directions_are_non_negative_as_the_catalogue_requires() -> None:
    spec = _spec(
        metric="grid.import_power_w",
        data_type="int16",
        split_negative_to="grid.export_power_w",
        asset_role="grid_connection",
    )
    for word in (500, 0xFE0C, 0):
        decoded = decode_register(spec, [word])
        primary = md.primary_value(spec, decoded)
        split = md.split_signed(spec, decoded)
        assert primary is not None and primary >= 0
        assert split is not None and split[1] >= 0


def test_a_register_without_a_split_passes_its_signed_value_through() -> None:
    spec = _spec(metric="battery.power_w", data_type="int16")
    decoded = decode_register(spec, [0xFE0C])
    assert md.primary_value(spec, decoded) == -500.0
    assert md.split_signed(spec, decoded) is None


def test_split_of_a_refused_reading_emits_nothing() -> None:
    spec = _spec(
        metric="grid.import_power_w",
        data_type="int16",
        split_negative_to="grid.export_power_w",
        asset_role="grid_connection",
        na_values=(0x8000,),
    )
    decoded = decode_register(spec, [0x8000])
    assert decoded.refusal == md.REFUSAL_NA_SENTINEL
    assert md.primary_value(spec, decoded) is None
    assert md.split_signed(spec, decoded) is None


# ── Adapter integration: framing, attribution, address base ─────
#
# The decoders above are pure. These exercise the seam where a vendor map
# reaches pymodbus: what framing the client is built with, which asset each
# reading is attributed to, and that no write path exists.

import textwrap  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

from collector.adapters import vendor_map as vm  # noqa: E402
from collector.adapters.generic_modbus import DEVICE_KWARG, GenericModbusAdapter  # noqa: E402

INVERTER_ID = "20000000-0000-0000-0000-000000000001"
BATTERY_ID = "20000000-0000-0000-0000-000000000002"
GRID_ID = "20000000-0000-0000-0000-000000000003"

ADAPTER_MAP = textwrap.dedent(
    """
    vendor: testvendor
    model_family: framing
    source: synthetic fixture (unit test)
    verification: community
    protocol_defaults:
      protocol: rtu
      baudrate: 115200
      parity: E
      stopbits: 2
      bytesize: 7
      unit_id: 10
      function_code: 3
      address_base: 1
    identity:
      serial: {address: 4, type: string, words: 2}
    assets: [inverter, battery, grid_connection]
    groups:
      - name: power
        poll_interval_seconds: 30
        registers:
          - {id: inv_w, metric: inverter.ac_output_power_w, address: 101,
             type: int16, asset: inverter}
          - {id: bat_soc, metric: battery.soc_pct, address: 102, type: uint16,
             asset: battery}
          - {id: grid_w, metric: grid.import_power_w, address: 103, type: int16,
             asset: grid_connection, split_negative_to: grid.export_power_w}
    """
)


def _install_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    (tmp_path / "framing_map.yaml").write_text(text)
    monkeypatch.setattr(vm, "VENDOR_MAPS_DIR", tmp_path)


def _adapter_config(**extra: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "vendor_map": "framing_map",
        "port": "/dev/ttyUSB0",
        "asset_mappings": {
            "inverter": INVERTER_ID,
            "battery": BATTERY_ID,
            "grid_connection": GRID_ID,
        },
    }
    config.update(extra)
    return config


def _serial_client(words_by_address: dict[int, list[int]]) -> AsyncMock:
    """A register-accurate fake: one WIRE address -> one word.

    Values are given per starting address (a multi-word register lists its
    words); the fake flattens them so any range the adapter asks for is served
    exactly as a real device would, and a range touching an address the device
    does not implement errors the whole request — which is how a real Modbus
    exception response behaves, and what the per-register fallback is for.
    """
    flat: dict[int, int] = {}
    for address, registers in words_by_address.items():
        for offset, word in enumerate(registers):
            flat[address + offset] = word

    client = AsyncMock()
    client.connect = AsyncMock(return_value=True)
    client.connected = True
    client.close = MagicMock()

    # pymodbus renamed slave -> device_id; the adapter passes whichever the
    # installed version accepts, so the fake accepts both.
    async def _read(address: int, count: int, **unit: int) -> MagicMock:
        result = MagicMock()
        try:
            registers = [flat[address + offset] for offset in range(count)]
        except KeyError:
            result.isError.return_value = True
            return result
        result.isError.return_value = False
        result.registers = registers
        return result

    client.read_holding_registers = AsyncMock(side_effect=_read)
    client.read_input_registers = AsyncMock(side_effect=_read)
    return client


@pytest.mark.asyncio
async def test_map_framing_reaches_the_serial_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DSE wants 115200 and some ComAp installs 8E1 — never assumed 8N1."""
    _install_map(tmp_path, monkeypatch, ADAPTER_MAP)
    adapter = GenericModbusAdapter()
    with patch(
        "collector.adapters.generic_modbus.AsyncModbusSerialClient"
    ) as MockClient:
        MockClient.return_value = _serial_client({})
        assert await adapter.initialize(_adapter_config()) is True

    kwargs = MockClient.call_args.kwargs
    assert kwargs["baudrate"] == 115200
    assert kwargs["parity"] == "E"
    assert kwargs["stopbits"] == 2
    assert kwargs["bytesize"] == 7


@pytest.mark.asyncio
async def test_site_settings_override_map_framing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_map(tmp_path, monkeypatch, ADAPTER_MAP)
    adapter = GenericModbusAdapter()
    with patch(
        "collector.adapters.generic_modbus.AsyncModbusSerialClient"
    ) as MockClient:
        MockClient.return_value = _serial_client({})
        await adapter.initialize(_adapter_config(baudrate=9600, parity="N"))

    kwargs = MockClient.call_args.kwargs
    assert kwargs["baudrate"] == 9600
    assert kwargs["parity"] == "N"


@pytest.mark.asyncio
async def test_each_register_is_attributed_to_its_own_bound_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One hybrid inverter, three assets — ingest rejects the alternative."""
    _install_map(tmp_path, monkeypatch, ADAPTER_MAP)
    adapter = GenericModbusAdapter()
    # address_base is 1, so a map address of 101 is read at wire address 100.
    words = {100: [2500], 101: [88], 102: [0xFE0C]}  # 2500 W, 88 %, -500 W
    with patch(
        "collector.adapters.generic_modbus.AsyncModbusSerialClient"
    ) as MockClient:
        MockClient.return_value = _serial_client(words)
        await adapter.initialize(_adapter_config())
        readings = await adapter.poll()

    by_metric = {r.metric_name: r for r in readings}
    assert by_metric["inverter.ac_output_power_w"].asset_id == INVERTER_ID
    assert by_metric["inverter.ac_output_power_w"].value == 2500.0
    assert by_metric["battery.soc_pct"].asset_id == BATTERY_ID
    assert by_metric["battery.soc_pct"].value == 88.0
    # The signed CT register feeds both directions on the grid asset.
    assert by_metric["grid.import_power_w"].value == 0.0
    assert by_metric["grid.export_power_w"].value == 500.0
    assert by_metric["grid.export_power_w"].asset_id == GRID_ID


@pytest.mark.asyncio
async def test_address_base_one_is_subtracted_once_before_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 1-based vendor table is transcribed as printed; the adapter adjusts."""
    _install_map(tmp_path, monkeypatch, ADAPTER_MAP)
    adapter = GenericModbusAdapter()
    client = _serial_client({100: [2500], 101: [88], 102: [0]})
    with patch(
        "collector.adapters.generic_modbus.AsyncModbusSerialClient"
    ) as MockClient:
        MockClient.return_value = client
        await adapter.initialize(_adapter_config())
        await adapter.poll()

    calls = [
        (call.kwargs["address"], call.kwargs["count"])
        for call in client.read_holding_registers.call_args_list
    ]
    # Printed 101-103 are contiguous, so they ride in one block starting at
    # wire address 100 — never at the printed address.
    assert (100, 3) in calls
    assert all(address != 101 or count != 1 for address, count in calls)


@pytest.mark.asyncio
async def test_registers_carry_the_maps_unit_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_map(tmp_path, monkeypatch, ADAPTER_MAP)
    adapter = GenericModbusAdapter()
    client = _serial_client({100: [1], 101: [1], 102: [1]})
    with patch(
        "collector.adapters.generic_modbus.AsyncModbusSerialClient"
    ) as MockClient:
        MockClient.return_value = client
        await adapter.initialize(_adapter_config())
        await adapter.poll()

    slaves = {
        call.kwargs[DEVICE_KWARG]
        for call in client.read_holding_registers.call_args_list
    }
    assert slaves == {10}


@pytest.mark.asyncio
async def test_identity_is_read_once_at_initialize_and_is_not_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_map(tmp_path, monkeypatch, ADAPTER_MAP)
    adapter = GenericModbusAdapter()
    # "AB" + "CD" at wire address 3 (printed 4, base 1)
    client = _serial_client({3: [0x4142, 0x4344], 100: [1], 101: [1], 102: [1]})
    with patch(
        "collector.adapters.generic_modbus.AsyncModbusSerialClient"
    ) as MockClient:
        MockClient.return_value = client
        await adapter.initialize(_adapter_config())
        reads_after_init = len(client.read_holding_registers.call_args_list)
        readings = await adapter.poll()

    assert adapter.identity() == {"serial": "ABCD"}
    assert reads_after_init == 1  # identity only
    assert not any(r.metric_name.startswith("identity.") for r in readings)
    assert all(isinstance(r.value, float) for r in readings)


@pytest.mark.asyncio
async def test_a_failed_identity_read_does_not_stop_the_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline-first: an unreadable serial register is not a reason to stop."""
    _install_map(tmp_path, monkeypatch, ADAPTER_MAP)
    adapter = GenericModbusAdapter()
    with patch(
        "collector.adapters.generic_modbus.AsyncModbusSerialClient"
    ) as MockClient:
        MockClient.return_value = _serial_client({100: [2500], 101: [88], 102: [0]})
        assert await adapter.initialize(_adapter_config()) is True
        readings = await adapter.poll()

    assert adapter.identity() == {}
    assert len(readings) == 4  # three registers, one of which splits


@pytest.mark.asyncio
async def test_a_read_error_on_one_register_does_not_lose_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_map(tmp_path, monkeypatch, ADAPTER_MAP)
    adapter = GenericModbusAdapter()
    with patch(
        "collector.adapters.generic_modbus.AsyncModbusSerialClient"
    ) as MockClient:
        # Wire 101 is unimplemented: the block read fails and the per-register
        # fallback recovers everything on either side of the hole.
        MockClient.return_value = _serial_client({100: [2500], 102: [0]})
        await adapter.initialize(_adapter_config())
        readings = await adapter.poll()

    metrics = {r.metric_name for r in readings}
    assert "inverter.ac_output_power_w" in metrics
    assert "battery.soc_pct" not in metrics


@pytest.mark.asyncio
async def test_a_vendor_map_adapter_declares_no_write_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_map(tmp_path, monkeypatch, ADAPTER_MAP)
    adapter = GenericModbusAdapter()
    with patch(
        "collector.adapters.generic_modbus.AsyncModbusSerialClient"
    ) as MockClient:
        MockClient.return_value = _serial_client({})
        await adapter.initialize(_adapter_config())

    caps = adapter.capabilities()
    assert caps.can_write is False
    assert caps.supported_write_metrics == []
    assert caps.supports_asset_targeting is True
    assert await adapter.write("battery.soc_pct", 50.0) is False


@pytest.mark.asyncio
async def test_a_map_paired_with_write_mappings_refuses_to_initialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse the combination rather than silently ignoring the writes."""
    _install_map(tmp_path, monkeypatch, ADAPTER_MAP)
    adapter = GenericModbusAdapter()
    config = _adapter_config(
        write_mappings={"battery.max_charge_w": {"address": 100, "type": "uint16"}}
    )
    with patch(
        "collector.adapters.generic_modbus.AsyncModbusSerialClient"
    ) as MockClient:
        MockClient.return_value = _serial_client({})
        assert await adapter.initialize(config) is False
    MockClient.assert_not_called()


@pytest.mark.asyncio
async def test_an_unresolvable_map_refuses_to_initialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_map(tmp_path, monkeypatch, ADAPTER_MAP)
    adapter = GenericModbusAdapter()
    config = _adapter_config(
        asset_mappings={"inverter": INVERTER_ID}
    )  # battery unbound
    with patch(
        "collector.adapters.generic_modbus.AsyncModbusSerialClient"
    ) as MockClient:
        MockClient.return_value = _serial_client({})
        assert await adapter.initialize(config) is False
    MockClient.assert_not_called()


@pytest.mark.asyncio
async def test_capabilities_report_the_maps_shortest_group_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Appended at the map's own indentation — groups sit two spaces in.
    text = ADAPTER_MAP + (
        "  - name: energy\n"
        "    poll_interval_seconds: 300\n"
        "    registers:\n"
        "      - {id: bat_kwh, metric: battery.charge_energy_kwh, address: 110,\n"
        "         type: uint32_le, scale: 0.1, asset: battery}\n"
    )
    _install_map(tmp_path, monkeypatch, text)
    adapter = GenericModbusAdapter()
    with patch(
        "collector.adapters.generic_modbus.AsyncModbusSerialClient"
    ) as MockClient:
        MockClient.return_value = _serial_client({})
        await adapter.initialize(_adapter_config())

    assert adapter.capabilities().poll_interval_seconds == 30
    assert "battery.charge_energy_kwh" in adapter.capabilities().supported_metrics
    assert "grid.export_power_w" in adapter.capabilities().supported_metrics
