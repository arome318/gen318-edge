"""Auto-discovering SunSpec Modbus telemetry adapter.

Unlike ``generic_modbus``, this adapter does not require a vendor register
map. It validates the ``SunS`` signature, walks the standard model chain and
decodes the common inverter (101/102/103) and meter (201/202/203) models with
their published scale factors. Unknown models are safely skipped by their
declared length, so vendor extensions cannot desynchronise discovery.

This is a client implementation, not a claim that attached hardware is
SunSpec Certified. Certification remains a property of the device/vendor.
"""

from __future__ import annotations

import math
import struct
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog

from .base import AdapterCapabilities, AdapterStatus, Reading
from .generic_modbus import DEVICE_KWARG, GenericModbusAdapter

logger = structlog.get_logger()

_SUNS_SIGNATURE = (0x5375, 0x6E53)
_END_MODEL = 0xFFFF
_INVERTER_MODELS = {101, 102, 103}
_METER_MODELS = {201, 202, 203}
# Float variants. Fronius and others let the installer choose "float" or
# "int+SF" at commissioning, and the adapter has to read whichever was picked
# — the model chain says which is present, so nothing is inferred.
_INVERTER_FLOAT_MODELS = {111, 112, 113}
_METER_FLOAT_MODELS = {211, 212, 213}
_STORAGE_MODEL = 124  # Basic Storage Controls, read-only here
_MPPT_MODEL = 160  # Multiple MPPT Inverter Extension
_NAMEPLATE_MODEL = 120
_SETTINGS_MODEL = 121
_IDENTITY_MODELS = {_NAMEPLATE_MODEL, _SETTINGS_MODEL}
_SUPPORTED_MODELS = (
    _INVERTER_MODELS
    | _METER_MODELS
    | _INVERTER_FLOAT_MODELS
    | _METER_FLOAT_MODELS
    | {_STORAGE_MODEL, _MPPT_MODEL}
)

# Offsets below include the two-register ID/L header, so the first data point
# is at offset 2. Every one is taken from the SunSpec Alliance's own model
# definitions (github.com/sunspec/models @ 90b4a331, Apache-2.0) — see
# docs/vendor-integrations/SOURCES.md.
_FLOAT_NAN = 0x7FC00000  # "not implemented" for float32

# Model 111/112/113 share one layout; only the mandatory flags differ.
_FLOAT_INVERTER_POINTS: tuple[tuple[str, int, str], ...] = (
    ("inverter.ac_output_current_a", 2, "A"),
    ("inverter.ac_output_voltage_v", 16, "PhVphA"),
    ("inverter.ac_output_power_w", 22, "W"),
    ("inverter.ac_output_freq_hz", 24, "Hz"),
    ("inverter.temperature_c", 40, "TmpCab"),
)
_FLOAT_INVERTER_PHASES: tuple[tuple[str, int, int, str], ...] = (
    ("l1", 4, 16, "phA"),
    ("l2", 6, 18, "phB"),
    ("l3", 8, 20, "phC"),
)
# Model 211/212/213 share one layout.
_FLOAT_METER_PHASES: tuple[tuple[str, int, int, int, str], ...] = (
    ("l1", 4, 12, 30, "phA"),
    ("l2", 6, 14, 32, "phB"),
    ("l3", 8, 16, 34, "phC"),
)
_FLOAT_METER_A = 2
_FLOAT_METER_PHV = 10
_FLOAT_METER_HZ = 26
_FLOAT_METER_W = 28
_FLOAT_METER_PF = 52
_FLOAT_METER_EXP = 60
_FLOAT_METER_IMP = 68

# Model 160: an 8-register fixed header from offset 2, then N modules of
# exactly 20 registers each starting at offset 10. One shared set of scale
# factors for the whole model, not one per module.
_MPPT_SF_DCA, _MPPT_SF_DCV, _MPPT_SF_DCW = 2, 3, 4
_MPPT_COUNT = 8
_MPPT_FIRST_MODULE = 10
_MPPT_MODULE_WORDS = 20
_MPPT_DCA, _MPPT_DCV, _MPPT_DCW = 9, 10, 11
#: solar.mppt{n}_* exists for n in 1..8 in the catalog.
_MPPT_MAX_MODULES = 8
_MAX_MODELS = 64
_MAX_MODEL_REGISTERS = 512
_MAX_READ_REGISTERS = 120


@dataclass(frozen=True, slots=True)
class ModelBlock:
    model_id: int
    address: int
    length: int


def _signed16(raw: int) -> int:
    return raw - 0x10000 if raw & 0x8000 else raw


def _scaled(
    registers: list[int],
    value_offset: int,
    scale_offset: int,
    *,
    signed: bool,
) -> float | None:
    raw = registers[value_offset]
    scale_raw = registers[scale_offset]
    unavailable = 0x8000 if signed else 0xFFFF
    if raw == unavailable or scale_raw == 0x8000:
        return None
    scale = _signed16(scale_raw)
    if scale < -10 or scale > 10:
        return None
    value = _signed16(raw) if signed else raw
    return float(value) * (10.0**scale)


def _scaled_acc32(
    registers: list[int], value_offset: int, scale_offset: int
) -> float | None:
    raw = (registers[value_offset] << 16) | registers[value_offset + 1]
    scale_raw = registers[scale_offset]
    if raw == 0xFFFFFFFF or scale_raw == 0x8000:
        return None
    scale = _signed16(scale_raw)
    if scale < -10 or scale > 10:
        return None
    return float(raw) * (10.0**scale)


class SunSpecModbusAdapter(GenericModbusAdapter):
    """SunSpec signature/model discovery over Modbus TCP or RTU."""

    def __init__(self) -> None:
        super().__init__()
        self._models: list[ModelBlock] = []
        self._unit_id = 1
        self._inverter_asset_id = ""
        self._meter_asset_id = ""
        # Storage (124) and MPPT (160) describe different assets from the
        # inverter that hosts them, and ingest rejects a battery metric sent
        # to an inverter asset. Unbound means those models are discovered and
        # logged but not emitted, rather than mis-addressed.
        self._battery_asset_id = ""
        self._pv_asset_id = ""
        self._nameplate: dict[str, float] = {}
        # asset_mappings: {"pcc": true} marks the meter as the utility
        # boundary — energies become grid.import/export, signed power becomes
        # the system grid view. Default False keeps customer-meter semantics.
        self._pcc = False

    async def initialize(self, config: dict[str, Any]) -> bool:
        if not await super().initialize({**config, "registers": []}):
            return False
        try:
            self._unit_id = int(config.get("unit_id", 1))
            if not 0 <= self._unit_id <= 247:
                raise ValueError("unit_id must be in the Modbus range 0..247")
        except (TypeError, ValueError):
            logger.error("sunspec_invalid_unit_id", unit_id=config.get("unit_id"))
            await self.shutdown()
            self._status = AdapterStatus.ERROR
            return False
        mappings = config.get("asset_mappings", {})
        if not isinstance(mappings, dict):
            mappings = {}
        default_asset = str(config.get("asset_id", ""))
        self._inverter_asset_id = str(mappings.get("inverter", default_asset))
        self._meter_asset_id = str(mappings.get("meter", default_asset))
        self._battery_asset_id = str(mappings.get("battery", ""))
        self._pv_asset_id = str(mappings.get("solar_array", ""))
        self._pcc = bool(mappings.get("pcc", False))

        configured_bases = config.get("base_addresses", [40000, 0, 50000])
        try:
            if not isinstance(configured_bases, (list, tuple)) or not configured_bases:
                raise ValueError("base_addresses must be a non-empty list")
            bases = [int(base) for base in configured_bases][:_MAX_MODELS]
            if any(base < 0 or base > 65533 for base in bases):
                raise ValueError("base address is outside the Modbus register space")
        except (TypeError, ValueError):
            logger.error("sunspec_invalid_base_addresses", value=configured_bases)
            await self.shutdown()
            self._status = AdapterStatus.ERROR
            return False
        for base in bases:
            try:
                signature = await self._read_words(base, 2)
                if tuple(signature) == _SUNS_SIGNATURE:
                    self._models = await self._discover_models(base + 2)
                    break
            except Exception:
                logger.debug("sunspec_base_probe_failed", base=base, exc_info=True)

        supported = [
            block for block in self._models if block.model_id in _SUPPORTED_MODELS
        ]
        if not supported:
            logger.error(
                "sunspec_no_supported_models",
                discovered=[block.model_id for block in self._models],
            )
            await self.shutdown()
            self._status = AdapterStatus.ERROR
            return False
        missing_mapping = (
            any(block.model_id in _INVERTER_MODELS for block in supported)
            and not self._valid_asset_id(self._inverter_asset_id)
        ) or (
            any(block.model_id in _METER_MODELS for block in supported)
            and not self._valid_asset_id(self._meter_asset_id)
        )
        if missing_mapping:
            logger.error(
                "sunspec_asset_mapping_required",
                inverter_asset_id=self._inverter_asset_id,
                meter_asset_id=self._meter_asset_id,
            )
            await self.shutdown()
            self._status = AdapterStatus.ERROR
            return False
        await self._read_nameplate()
        logger.info(
            "sunspec_discovered",
            models=[block.model_id for block in self._models],
            supported=[block.model_id for block in supported],
            nameplate=self._nameplate,
        )
        return True

    async def _read_nameplate(self) -> None:
        """Models 120/121 once at initialize — ratings, never telemetry.

        A validation record needs to say what the unit claims it is. Note the
        Alliance's definition has no bare VRtg point in model 120, only VARtg;
        the spec asked for "WRtg, VRtg" and only one of those exists.
        """
        for block in self._models:
            if block.model_id not in _IDENTITY_MODELS:
                continue
            try:
                registers = await self._read_words(block.address, block.length + 2)
            except Exception:
                logger.info("sunspec_nameplate_unavailable", model=block.model_id)
                continue
            if block.model_id == _NAMEPLATE_MODEL and len(registers) > 6:
                rated_w = _scaled(registers, 3, 4, signed=False)
                rated_va = _scaled(registers, 5, 6, signed=False)
                if rated_w is not None:
                    self._nameplate["rated_w"] = rated_w
                if rated_va is not None:
                    self._nameplate["rated_va"] = rated_va

    def nameplate(self) -> dict[str, float]:
        """Ratings read from model 120, or empty. Never telemetry."""
        return dict(self._nameplate)

    async def _read_words(self, address: int, count: int) -> list[int]:
        if self._client is None:
            raise RuntimeError("SunSpec client is not connected")
        if count < 1 or address < 0 or address + count > 65536:
            raise ValueError(f"invalid Modbus register range {address}+{count}")
        words: list[int] = []
        cursor = address
        remaining = count
        while remaining:
            chunk = min(remaining, _MAX_READ_REGISTERS)
            result = await self._client.read_holding_registers(
                address=cursor,
                count=chunk,
                **{DEVICE_KWARG: self._unit_id},
            )
            if result.isError() or len(result.registers) != chunk:
                raise OSError(f"SunSpec Modbus read failed at {cursor}")
            words.extend(int(register) for register in result.registers)
            cursor += chunk
            remaining -= chunk
        return words

    async def _discover_models(self, cursor: int) -> list[ModelBlock]:
        models: list[ModelBlock] = []
        for _ in range(_MAX_MODELS):
            if cursor < 0 or cursor > 65534:
                raise ValueError("SunSpec model chain exceeds Modbus register space")
            model_id, length = await self._read_words(cursor, 2)
            if model_id == _END_MODEL:
                return models
            if length <= 0 or length > _MAX_MODEL_REGISTERS:
                raise ValueError(f"invalid SunSpec model length {length} at {cursor}")
            models.append(ModelBlock(model_id=model_id, address=cursor, length=length))
            cursor += length + 2
        raise ValueError("SunSpec model chain exceeds safety limit")

    @staticmethod
    def _valid_asset_id(value: str) -> bool:
        try:
            uuid.UUID(value)
        except (AttributeError, TypeError, ValueError):
            return False
        return True

    async def poll(self) -> list[Reading]:
        if not self._client or not self._client.connected:
            return []
        now = datetime.now(UTC)
        readings: list[Reading] = []
        try:
            for block in self._models:
                if block.model_id not in _SUPPORTED_MODELS:
                    continue
                registers = await self._read_words(block.address, block.length + 2)
                if registers[0] != block.model_id or registers[1] != block.length:
                    raise ValueError(f"SunSpec model header changed at {block.address}")
                if block.model_id in _INVERTER_MODELS:
                    readings.extend(self._decode_inverter(registers, block, now))
                elif block.model_id in _INVERTER_FLOAT_MODELS:
                    readings.extend(self._decode_float_inverter(registers, block, now))
                elif block.model_id in _METER_FLOAT_MODELS:
                    readings.extend(self._decode_float_meter(registers, block, now))
                elif block.model_id == _STORAGE_MODEL:
                    readings.extend(self._decode_storage(registers, block, now))
                elif block.model_id == _MPPT_MODEL:
                    readings.extend(self._decode_mppt(registers, block, now))
                else:
                    readings.extend(self._decode_meter(registers, block, now))
        except Exception:
            self._error_count += 1
            self._status = AdapterStatus.ERROR
            logger.warning("sunspec_poll_failed", exc_info=True)
            return []

        if readings:
            self._last_reading_at = time.time()
            self._error_count = 0
            self._status = AdapterStatus.CONNECTED
        return readings

    def _decode_inverter(
        self, registers: list[int], block: ModelBlock, now: datetime
    ) -> list[Reading]:
        # Offsets include the two-register ID/L header and are common across
        # SunSpec models 101-103; mandatory/optional phase points differ.
        points = (
            ("inverter.ac_output_current_a", 2, 6, False, "A"),
            ("inverter.ac_output_voltage_v", 10, 13, False, "PhVphA"),
            ("inverter.ac_output_power_w", 14, 15, True, "W"),
            ("inverter.ac_output_freq_hz", 16, 17, False, "Hz"),
            ("inverter.temperature_c", 33, 37, True, "TmpCab"),
        )
        readings: list[Reading] = []
        for metric, value_offset, scale_offset, signed, point in points:
            if max(value_offset, scale_offset) >= len(registers):
                continue
            value = _scaled(registers, value_offset, scale_offset, signed=signed)
            if value is not None:
                readings.append(
                    self._reading(
                        metric, value, block, point, now, self._inverter_asset_id
                    )
                )
        for metric, offset, size, point in (
            ("inverter.state", 38, 1, "St"),
            ("inverter.error_code", 40, 2, "Evt1"),
        ):
            if offset + size > len(registers):
                continue
            raw = (
                registers[offset]
                if size == 1
                else (registers[offset] << 16) | registers[offset + 1]
            )
            unavailable = 0xFFFF if size == 1 else 0xFFFFFFFF
            if raw != unavailable:
                readings.append(
                    self._reading(
                        metric,
                        float(raw),
                        block,
                        point,
                        now,
                        self._inverter_asset_id,
                    )
                )
        return readings

    def _decode_meter(
        self, registers: list[int], block: ModelBlock, now: datetime
    ) -> list[Reading]:
        """Decode a 20x meter block.

        Customer meter (default): today's meter.* names plus per-phase detail.
        PCC meter (``asset_mappings: {"pcc": true}``): voltage/frequency
        describe the grid, energies are grid.import/export (TotWhImp at
        offset 46 was previously never read — backwards for an
        import-dominant C&I site), and the signed total power is the system
        grid view rather than a site meter reading.
        """
        points = (
            ("meter.current_a", 2, 6, "A"),
            ("grid.voltage_v" if self._pcc else "meter.voltage_v", 7, 15, "PhV"),
            ("system.grid_power_w" if self._pcc else "meter.power_w", 18, 22, "W"),
            ("meter.power_factor", 33, 37, "PF"),
        )
        readings: list[Reading] = []
        for metric, value_offset, scale_offset, point in points:
            if max(value_offset, scale_offset) >= len(registers):
                continue
            value = _scaled(registers, value_offset, scale_offset, signed=True)
            if value is None:
                continue
            if metric == "meter.power_factor":
                value /= 100.0
            readings.append(
                self._reading(metric, value, block, point, now, self._meter_asset_id)
            )

        # Per-phase detail (both modes) — AphX/PhVphX/WphX share the family
        # scale factors at 6/15/22.
        phase_points = (
            ("l1", 3, 8, 19, "phA"),
            ("l2", 4, 9, 20, "phB"),
            ("l3", 5, 10, 21, "phC"),
        )
        for phase, a_off, v_off, w_off, point_suffix in phase_points:
            if max(a_off, v_off, w_off, 22) >= len(registers):
                continue
            current = _scaled(registers, a_off, 6, signed=True)
            if current is not None:
                readings.append(
                    self._reading(
                        f"meter.{phase}_current_a",
                        current,
                        block,
                        f"A{point_suffix}",
                        now,
                        self._meter_asset_id,
                    )
                )
            voltage = _scaled(registers, v_off, 15, signed=True)
            if voltage is not None:
                readings.append(
                    self._reading(
                        f"meter.{phase}_voltage_v",
                        voltage,
                        block,
                        f"PhV{point_suffix}",
                        now,
                        self._meter_asset_id,
                    )
                )
            power = _scaled(registers, w_off, 22, signed=True)
            if power is not None:
                readings.append(
                    self._reading(
                        f"meter.{phase}_power_w",
                        power,
                        block,
                        f"W{point_suffix}",
                        now,
                        self._meter_asset_id,
                    )
                )

        if self._pcc and len(registers) > 17:
            frequency = _scaled(registers, 16, 17, signed=False)
            if frequency is not None:
                readings.append(
                    self._reading(
                        "grid.frequency_hz",
                        frequency,
                        block,
                        "Hz",
                        now,
                        self._meter_asset_id,
                    )
                )

        if len(registers) > 54:
            export_wh = _scaled_acc32(registers, 38, 54)
            if export_wh is not None:
                readings.append(
                    self._reading(
                        "grid.export_energy_kwh" if self._pcc else "meter.energy_kwh",
                        export_wh / 1000.0,
                        block,
                        "TotWhExp",
                        now,
                        self._meter_asset_id,
                    )
                )
            if self._pcc:
                import_wh = _scaled_acc32(registers, 46, 54)
                if import_wh is not None:
                    readings.append(
                        self._reading(
                            "grid.import_energy_kwh",
                            import_wh / 1000.0,
                            block,
                            "TotWhImp",
                            now,
                            self._meter_asset_id,
                        )
                    )
        return readings

    @staticmethod
    def _float32(registers: list[int], offset: int) -> float | None:
        """Decode a float32 point, or None when it is not implemented.

        SunSpec float32 puts the HIGH word at the LOWER address and signals
        "not implemented" with a quiet NaN. NaN is refused rather than passed
        on: it survives arithmetic silently and poisons any average built from
        it downstream.
        """
        if offset + 1 >= len(registers):
            return None
        raw = (registers[offset] << 16) | registers[offset + 1]
        if raw == _FLOAT_NAN:
            return None
        value = float(struct.unpack(">f", struct.pack(">I", raw))[0])
        return value if math.isfinite(value) else None

    def _decode_float_inverter(
        self, registers: list[int], block: ModelBlock, now: datetime
    ) -> list[Reading]:
        """Models 111/112/113 — the same points as 101/103 without scale factors."""
        readings: list[Reading] = []
        for metric, offset, point in _FLOAT_INVERTER_POINTS:
            value = self._float32(registers, offset)
            if value is not None:
                readings.append(
                    self._reading(
                        metric, value, block, point, now, self._inverter_asset_id
                    )
                )
        for phase, current_offset, voltage_offset, point in _FLOAT_INVERTER_PHASES:
            current = self._float32(registers, current_offset)
            if current is not None:
                readings.append(
                    self._reading(
                        f"inverter.ac_output_{phase}_current_a",
                        current,
                        block,
                        f"Aph{point[-1]}",
                        now,
                        self._inverter_asset_id,
                    )
                )
            voltage = self._float32(registers, voltage_offset)
            if voltage is not None:
                readings.append(
                    self._reading(
                        f"inverter.ac_output_{phase}_voltage_v",
                        voltage,
                        block,
                        f"PhV{point}",
                        now,
                        self._inverter_asset_id,
                    )
                )
        # St and Evt1 are integers even in the float models.
        for metric, offset, size, point in (
            ("inverter.state", 48, 1, "St"),
            ("inverter.error_code", 50, 2, "Evt1"),
        ):
            if offset + size > len(registers):
                continue
            raw = (
                registers[offset]
                if size == 1
                else (registers[offset] << 16) | registers[offset + 1]
            )
            if raw != (0xFFFF if size == 1 else 0xFFFFFFFF):
                readings.append(
                    self._reading(
                        metric, float(raw), block, point, now, self._inverter_asset_id
                    )
                )
        return readings

    def _decode_float_meter(
        self, registers: list[int], block: ModelBlock, now: datetime
    ) -> list[Reading]:
        """Models 211/212/213 — PCC semantics identical to the integer models.

        The PCC treatment the integer path already has is preserved exactly:
        a meter flagged ``pcc`` reports grid.* energies and the system grid
        view, and a customer meter keeps the meter.* names.
        """
        readings: list[Reading] = []
        for metric, offset, point in (
            ("meter.current_a", _FLOAT_METER_A, "A"),
            (
                "grid.voltage_v" if self._pcc else "meter.voltage_v",
                _FLOAT_METER_PHV,
                "PhV",
            ),
            (
                "system.grid_power_w" if self._pcc else "meter.power_w",
                _FLOAT_METER_W,
                "W",
            ),
            ("meter.power_factor", _FLOAT_METER_PF, "PF"),
        ):
            value = self._float32(registers, offset)
            if value is None:
                continue
            if metric == "meter.power_factor":
                # The float models report PF as a percentage, as the integer
                # models do.
                value /= 100.0
            readings.append(
                self._reading(metric, value, block, point, now, self._meter_asset_id)
            )

        for phase, a_off, v_off, w_off, point in _FLOAT_METER_PHASES:
            for metric, offset, label in (
                (f"meter.{phase}_current_a", a_off, f"Aph{point[-1]}"),
                (f"meter.{phase}_voltage_v", v_off, f"PhV{point}"),
                (f"meter.{phase}_power_w", w_off, f"Wph{point[-1]}"),
            ):
                value = self._float32(registers, offset)
                if value is not None:
                    readings.append(
                        self._reading(
                            metric, value, block, label, now, self._meter_asset_id
                        )
                    )

        if self._pcc:
            frequency = self._float32(registers, _FLOAT_METER_HZ)
            if frequency is not None:
                readings.append(
                    self._reading(
                        "grid.frequency_hz",
                        frequency,
                        block,
                        "Hz",
                        now,
                        self._meter_asset_id,
                    )
                )

        export_wh = self._float32(registers, _FLOAT_METER_EXP)
        if export_wh is not None:
            readings.append(
                self._reading(
                    "grid.export_energy_kwh" if self._pcc else "meter.energy_kwh",
                    export_wh / 1000.0,
                    block,
                    "TotWhExp",
                    now,
                    self._meter_asset_id,
                )
            )
        if self._pcc:
            import_wh = self._float32(registers, _FLOAT_METER_IMP)
            if import_wh is not None:
                readings.append(
                    self._reading(
                        "grid.import_energy_kwh",
                        import_wh / 1000.0,
                        block,
                        "TotWhImp",
                        now,
                        self._meter_asset_id,
                    )
                )
        return readings

    def _decode_storage(
        self, registers: list[int], block: ModelBlock, now: datetime
    ) -> list[Reading]:
        """Model 124 — READ ONLY. State of charge, and nothing else.

        ChaState (offset 8, SF at 22) is the battery's reported state of
        charge. WChaMax and StorCtl_Mod are settable points and this adapter
        never writes; WChaMax is not emitted as telemetry either, because the
        only name for it in the vocabulary (battery.max_charge_w) is a WRITE
        metric deliberately excluded from ALL_METRICS, and ingest would reject
        it. Reporting a setpoint as a measurement is its own kind of wrong.
        """
        readings: list[Reading] = []
        if max(8, 22) >= len(registers):
            return readings
        soc = _scaled(registers, 8, 22, signed=False)
        if soc is not None and self._battery_asset_id:
            readings.append(
                self._reading(
                    "battery.soc_pct",
                    soc,
                    block,
                    "ChaState",
                    now,
                    self._battery_asset_id,
                )
            )
        return readings

    def _decode_mppt(
        self, registers: list[int], block: ModelBlock, now: datetime
    ) -> list[Reading]:
        """Model 160 — per-module DC current, voltage and power.

        One shared set of scale factors for the whole model. Modules are
        20 registers each from offset 10; ``N`` at offset 8 says how many,
        and the block's own declared length is the second bound so a wrong
        N cannot walk off the end of what was read.
        """
        readings: list[Reading] = []
        if len(registers) <= _MPPT_FIRST_MODULE:
            return readings
        declared = registers[8] if len(registers) > 8 else 0
        available = (len(registers) - _MPPT_FIRST_MODULE) // _MPPT_MODULE_WORDS
        modules = min(declared, available, _MPPT_MAX_MODULES)
        for index in range(modules):
            base = _MPPT_FIRST_MODULE + index * _MPPT_MODULE_WORDS
            for metric_suffix, point_offset, scale_offset, point in (
                ("current_a", _MPPT_DCA, _MPPT_SF_DCA, "DCA"),
                ("voltage_v", _MPPT_DCV, _MPPT_SF_DCV, "DCV"),
                ("power_w", _MPPT_DCW, _MPPT_SF_DCW, "DCW"),
            ):
                offset = base + point_offset
                if offset >= len(registers):
                    continue
                value = _scaled(registers, offset, scale_offset, signed=False)
                if value is None:
                    continue
                readings.append(
                    self._reading(
                        f"solar.mppt{index + 1}_{metric_suffix}",
                        value,
                        block,
                        f"{point}[{index}]",
                        now,
                        self._pv_asset_id or self._inverter_asset_id,
                    )
                )
        return readings

    @staticmethod
    def _reading(
        metric: str,
        value: float,
        block: ModelBlock,
        point: str,
        now: datetime,
        asset_id: str,
    ) -> Reading:
        return Reading(
            metric_name=metric,
            value=value,
            timestamp=now,
            quality=0,
            raw_register=f"SunSpec:{block.model_id}:{point}",
            asset_id=asset_id,
        )

    def capabilities(self) -> AdapterCapabilities:
        metrics = [
            "inverter.ac_output_current_a",
            "inverter.ac_output_voltage_v",
            "inverter.ac_output_power_w",
            "inverter.ac_output_freq_hz",
            "inverter.temperature_c",
            "inverter.state",
            "inverter.error_code",
            "meter.current_a",
            "meter.voltage_v",
            "meter.power_w",
            "meter.power_factor",
            "meter.energy_kwh",
            # Model 124 (read-only) and model 160.
            "battery.soc_pct",
            *[
                f"solar.mppt{index}_power_w"
                for index in range(1, _MPPT_MAX_MODULES + 1)
            ],
            *[
                f"solar.mppt{index}_voltage_v"
                for index in range(1, _MPPT_MAX_MODULES + 1)
            ],
            *[
                f"solar.mppt{index}_current_a"
                for index in range(1, _MPPT_MAX_MODULES + 1)
            ],
        ]
        return AdapterCapabilities(
            can_read=True,
            can_write=False,
            can_subscribe=False,
            supported_metrics=metrics,
            poll_interval_seconds=int(self._config.get("poll_interval_seconds", 60)),
            min_poll_interval_seconds=5,
        )
