"""YAML-configurable Modbus adapter for any Modbus TCP/RTU device (Doc 15 §7.6).

Two configuration styles, and both keep working:

* **Legacy** — ``registers:`` inline in adapters.yaml, one read per register,
  8N1 framing, big-endian 32-bit words, one ``asset_id`` for the instance.
  Byte-for-byte unchanged, including the installer-declared write path.
* **Vendor map** (vendor integrations spec §2) — ``vendor_map: <name>``
  resolves a shipped declarative map (see ``vendor_maps/``). One device then
  feeds SEVERAL assets: each register names a ROLE and the site YAML binds
  that role to an asset UUID, which is what a hybrid inverter needs, since
  ingest rejects a battery metric addressed to an inverter asset.

The map path adds configurable serial framing (7/8 bit, parity, stopbits),
per-register unit ids, 0/1-based address bases, and the decoders in
``modbus_decode`` — word-swapped 32-bit values, u64, enums, bitfields, fault
words, scale-then-offset and declared sentinels. A vendor map can never
write: the loader refuses write keys, and a site that pairs a map with
``write_mappings`` refuses to initialize rather than silently ignoring them.
"""

from __future__ import annotations

import asyncio
import inspect
import struct
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import structlog
from pymodbus.client import AsyncModbusSerialClient, AsyncModbusTcpClient
from pymodbus.exceptions import ModbusIOException
from pymodbus.framer import FramerType

from .base import AdapterCapabilities, AdapterStatus, HardwareAdapter, Reading
from .modbus_decode import (
    Decoded,
    decode_register,
    derive_sum,
    primary_value,
    split_signed,
)
from .vendor_map import (
    RegisterSpec,
    VendorMap,
    VendorMapError,
    resolve_vendor_map,
    vendor_map_name,
)

logger = structlog.get_logger()


def _device_kwarg() -> str:
    """The keyword this pymodbus calls the Modbus unit/slave id.

    pymodbus renamed ``slave`` to ``device_id`` in 3.11 and REMOVED the old
    name by 3.14, where passing it raises TypeError. ``requirements.txt`` pins
    only ``pymodbus>=3.6.0``, so a node built today resolves 3.14 and a node
    built last year resolves 3.9 — both are in the field.

    Every existing test mocks the client, so nothing catches this: the call
    succeeds against a MagicMock whatever the keyword is named. It only shows
    up against a real server, which is why the vendor-map tests run one.

    Resolved once at import against the installed signature, rather than
    guessed from a version string.
    """
    try:
        from pymodbus.client.mixin import ModbusClientMixin

        parameters = inspect.signature(
            ModbusClientMixin.read_holding_registers
        ).parameters
    except Exception:  # pragma: no cover - defensive: odd/absent pymodbus
        return "slave"
    return "device_id" if "device_id" in parameters else "slave"


#: Whichever name this pymodbus accepts. Used by the vendor-map read paths.
DEVICE_KWARG = _device_kwarg()


class GenericModbusAdapter(HardwareAdapter):
    """Reads any Modbus device using register definitions from YAML config.

    Supports both Modbus TCP and Modbus RTU protocols. Register mappings,
    scaling factors, and data types are fully configurable via YAML.
    """

    def __init__(self) -> None:
        self._client: AsyncModbusTcpClient | AsyncModbusSerialClient | None = None
        self._config: dict[str, Any] = {}
        self._status = AdapterStatus.DISCONNECTED
        self._last_reading_at: float = 0.0
        self._error_count: int = 0
        self._registers: list[dict[str, Any]] = []
        self._asset_id: str = ""
        # Vendor-map state. ``_map is None`` means the legacy path, which must
        # behave exactly as it did before maps existed.
        self._map: VendorMap | None = None
        self._specs: list[RegisterSpec] = []
        self._identity: dict[str, str] = {}
        # Monotonic deadline per register group. time.monotonic() cannot go
        # backwards, so an NTP step on a Pi that has just found the network
        # cannot make every group look overdue at once.
        self._group_due: dict[str, float] = {}
        self._max_gap = 8
        self._max_block_words = 120
        self._request_spacing_s = 0.0

    async def initialize(self, config: dict[str, Any]) -> bool:
        """Connect to the Modbus device (TCP or RTU)."""
        self._config = config
        self._registers = config.get("registers", [])
        self._asset_id = config.get("asset_id", "generic")

        transport = config
        map_name = None
        try:
            map_name = vendor_map_name(config)
            if map_name is not None:
                if config.get("write_mappings"):
                    raise VendorMapError(
                        "a vendor_map config cannot declare write_mappings — vendor "
                        "maps are read-only"
                    )
                # A validation sweep reads registers to produce evidence and
                # emits no telemetry, so it binds no assets.
                self._map = resolve_vendor_map(
                    map_name,
                    config,
                    require_bindings=not bool(config.get("probe_mode")),
                )
                self._specs = [
                    spec
                    for spec in self._map.all_registers
                    if spec.data_type != "string"
                ]
                # Map defaults already carry the site's overrides.
                transport = {**config, **self._map.protocol}
                self._max_gap = max(0, int(transport.get("max_gap", 8)))
                self._max_block_words = max(
                    1, min(125, int(transport.get("max_block_words", 120)))
                )
                # WiNet-S needs >=20 ms between requests and a ComAp answers
                # one object at a time; a shared RS485 bus is not a network.
                self._request_spacing_s = (
                    max(0.0, float(transport.get("request_spacing_ms", 0))) / 1000.0
                )
        except VendorMapError as exc:
            logger.error("vendor_map_rejected", vendor_map=map_name, error=str(exc))
            self._status = AdapterStatus.ERROR
            return False

        protocol = transport.get("protocol", "tcp")
        timeout_s = transport.get("timeout_ms", 3000) / 1000.0

        try:
            if protocol == "tcp":
                self._client = AsyncModbusTcpClient(
                    host=transport["host"],
                    port=transport.get("port", 502),
                    timeout=timeout_s,
                    retries=transport.get("retries", 3),
                    reconnect_delay=1,
                    reconnect_delay_max=60,
                    framer=FramerType.SOCKET,
                )
            elif protocol == "rtu":
                # DSE wants 115200, ComAp 57600, and some IG-NT installs run
                # 8E1 — framing is configurable, never assumed.
                self._client = AsyncModbusSerialClient(
                    port=transport["port"],
                    baudrate=transport.get("baudrate", 9600),
                    bytesize=int(transport.get("bytesize", 8)),
                    parity=str(transport.get("parity", "N")).upper(),
                    stopbits=int(transport.get("stopbits", 1)),
                    timeout=timeout_s,
                    retries=transport.get("retries", 3),
                )
            else:
                logger.error("unsupported_modbus_protocol", protocol=protocol)
                return False

            connected = await self._client.connect()
            self._status = AdapterStatus.CONNECTED if connected else AdapterStatus.ERROR
            logger.info(
                "generic_modbus_initialized",
                protocol=protocol,
                asset_id=self._asset_id,
                vendor_map=self._map.name if self._map else None,
                verification=self._map.verification if self._map else None,
                register_count=len(self._specs) if self._map else len(self._registers),
                connected=connected,
            )
            if connected and self._map is not None:
                await self._read_identity()
            return connected

        except Exception:
            logger.exception("generic_modbus_init_failed", protocol=protocol)
            self._status = AdapterStatus.ERROR
            return False

    async def poll(self) -> list[Reading]:
        """Read all configured registers."""
        if not self._client or not self._client.connected:
            return []
        if self._map is not None:
            return await self._poll_vendor_map(self._due_specs(time.monotonic()))

        readings: list[Reading] = []
        now = datetime.now(UTC)
        unit_id = self._config.get("unit_id", 1)

        for reg_def in self._registers:
            try:
                value = await self._read_register(unit_id, reg_def)
                if value is not None:
                    readings.append(
                        Reading(
                            metric_name=reg_def["metric"],
                            value=value,
                            timestamp=now,
                            quality=0,
                            raw_register=f"modbus:{unit_id}:{reg_def['address']}",
                            asset_id=self._asset_id,
                        )
                    )
            except ModbusIOException:
                self._error_count += 1
                logger.warning(
                    "generic_modbus_read_error",
                    unit_id=unit_id,
                    address=reg_def.get("address"),
                    metric=reg_def.get("metric"),
                )

        if readings:
            self._last_reading_at = time.time()
            self._error_count = 0

        return readings

    async def _read_register(
        self, unit_id: int, reg_def: dict[str, Any]
    ) -> float | None:
        """Read a single register using the configured function code."""
        assert self._client is not None

        address: int = reg_def["address"]
        data_type: str = reg_def.get("type", "uint16")
        scale: float = reg_def.get("scale", 1.0)
        func_code: int = reg_def.get("function_code", 3)
        count = 2 if data_type in ("uint32", "int32", "float32") else 1

        if func_code == 3:
            result = await self._client.read_holding_registers(
                address=address,
                count=count,
                **{DEVICE_KWARG: unit_id},
            )
        elif func_code == 4:
            result = await self._client.read_input_registers(
                address=address,
                count=count,
                **{DEVICE_KWARG: unit_id},
            )
        else:
            logger.warning("unsupported_function_code", code=func_code)
            return None

        if result.isError():
            return None

        raw = result.registers[0]

        if data_type == "int16":
            if raw > 32767:
                raw -= 65536
        elif data_type == "uint32":
            raw = (result.registers[0] << 16) | result.registers[1]
        elif data_type == "int32":
            raw = (result.registers[0] << 16) | result.registers[1]
            if raw > 2147483647:
                raw -= 4294967296
        elif data_type == "float32":
            raw_float = struct.unpack(
                ">f",
                struct.pack(">HH", result.registers[0], result.registers[1]),
            )[0]
            return float(raw_float) * scale

        return float(raw) * scale

    # ── Vendor-map path ──────────────────────────────────────────

    async def _read_words(self, spec: RegisterSpec, count: int) -> list[int] | None:
        """Read ``count`` words at a spec's address. None on any read failure.

        The map prints addresses as the vendor documents them; ``address_base``
        says whether that is 0- or 1-based and the subtraction happens here,
        once, so a transcriber can copy a 1-based table verbatim.
        """
        assert self._client is not None
        base = self._map.address_base if self._map else 0
        address = spec.address - base
        if address < 0 or count < 1 or address + count > 65536:
            logger.warning(
                "vendor_map_address_out_of_range",
                register=spec.id,
                address=spec.address,
                count=count,
            )
            return None
        try:
            unit = {DEVICE_KWARG: spec.unit_id}
            if spec.function_code == 4:
                result = await self._client.read_input_registers(
                    address=address, count=count, **unit
                )
            else:
                result = await self._client.read_holding_registers(
                    address=address, count=count, **unit
                )
        except (ModbusIOException, OSError, asyncio.TimeoutError):
            self._error_count += 1
            logger.warning(
                "vendor_map_read_error",
                register=spec.id,
                address=spec.address,
                unit_id=spec.unit_id,
            )
            return None
        if result.isError():
            return None
        return [int(word) for word in result.registers]

    async def _read_identity(self) -> None:
        """Read the identity block once, at initialize.

        Serial, model and firmware are what makes a validation record about a
        UNIT rather than about a brand — register maps drift by firmware, so a
        record with no identity cannot be checked against anything later. Never
        telemetry: identity strings never reach ``measurements``.
        """
        if self._map is None:
            return
        for spec in self._map.identity:
            words = await self._read_words(spec, spec.word_count)
            if words is None:
                logger.info("vendor_map_identity_unavailable", register=spec.id)
                continue
            decoded = decode_register(spec, words)
            if decoded.text is not None:
                self._identity[spec.id] = decoded.text
            elif decoded.value is not None:
                self._identity[spec.id] = str(decoded.value)
        if self._identity:
            logger.info(
                "vendor_map_identity",
                vendor_map=self._map.name,
                vendor=self._map.vendor,
                model_family=self._map.model_family,
                **self._identity,
            )

    def identity(self) -> dict[str, str]:
        """Identity fields read at initialize (empty on the legacy path)."""
        return dict(self._identity)

    def vendor_map_summary(self) -> dict[str, Any] | None:
        """The resolved map's provenance, or None on the legacy path.

        The collector ships this as one tier-3 event so a validation record
        can say which map revision, at which verification tier, produced a
        given series — register maps drift by firmware, and a record that
        cannot name the map it used proves nothing later.
        """
        if self._map is None:
            return None
        summary = self._map.to_summary()
        summary["identity"] = dict(self._identity)
        return summary

    def _due_specs(self, now: float) -> list[RegisterSpec]:
        """Registers from every group whose cadence has come round.

        Power every 30 s and energy counters every 300 s is the difference
        between a 9600-baud bus that keeps up and one that does not. A group
        that has fallen behind is rescheduled from NOW, never from its missed
        deadline: after an hour offline, catching up would fire a burst of
        reads at a device that has just come back.
        """
        if self._map is None:
            return []
        due: list[RegisterSpec] = []
        for group in self._map.groups:
            deadline = self._group_due.get(group.name)
            if deadline is None or now >= deadline:
                self._group_due[group.name] = now + group.poll_interval_seconds
                due.extend(group.registers)
        return due

    @staticmethod
    def _plan_blocks(
        specs: list[RegisterSpec], *, max_gap: int, max_words: int
    ) -> list[tuple[int, int, int, int, list[RegisterSpec]]]:
        """Group contiguous registers into one request each.

        Returns ``(unit_id, function_code, start, count, specs)``. A Deye poll
        of ~60 registers becomes ~6 requests, which is what makes the map
        viable on a shared 9600-baud line.

        Only registers sharing a unit id AND a function code can ride in one
        request — they are different conversations with different devices.
        Within that, a new block starts when the hole to the next register
        exceeds ``max_gap`` (reading 200 reserved words to save a request is
        not a saving) or when the span would exceed ``max_words`` (Modbus caps
        a read at 125; 120 is the house limit, and it counts the width of the
        last register, not just its address).
        """
        blocks: list[tuple[int, int, int, int, list[RegisterSpec]]] = []
        readable = [spec for spec in specs if not spec.derived]
        keyed: dict[tuple[int, int], list[RegisterSpec]] = {}
        for spec in readable:
            keyed.setdefault((spec.unit_id, spec.function_code), []).append(spec)

        for (unit_id, function_code), group in sorted(keyed.items()):
            ordered = sorted(group, key=lambda s: (s.address, s.word_count))
            current: list[RegisterSpec] = []
            start = 0
            end = 0  # exclusive, in printed-address space
            for spec in ordered:
                spec_end = spec.address + spec.word_count
                if not current:
                    current, start, end = [spec], spec.address, spec_end
                    continue
                gap = spec.address - end
                if gap > max_gap or (max(end, spec_end) - start) > max_words:
                    blocks.append((unit_id, function_code, start, end - start, current))
                    current, start, end = [spec], spec.address, spec_end
                else:
                    current.append(spec)
                    end = max(end, spec_end)
            if current:
                blocks.append((unit_id, function_code, start, end - start, current))
        return blocks

    async def _poll_vendor_map(self, specs: list[RegisterSpec]) -> list[Reading]:
        """Read the due registers as blocks and decode each from its slice."""
        if not specs:
            return []
        now = datetime.now(UTC)
        decoded_by_id: dict[str, Decoded] = {}
        readings: list[Reading] = []
        blocks = self._plan_blocks(
            specs, max_gap=self._max_gap, max_words=self._max_block_words
        )

        for index, (unit_id, function_code, start, count, block_specs) in enumerate(
            blocks
        ):
            if index:
                await self._pace()
            words = await self._read_block(unit_id, function_code, start, count)
            if words is None or len(words) < count:
                # A vendor that faults on one reserved address inside the span
                # would otherwise cost the whole block. Deye does exactly this.
                decoded_here = await self._read_block_individually(block_specs)
            else:
                decoded_here = {}
                for spec in block_specs:
                    offset = spec.address - start
                    decoded_here[spec.id] = decode_register(
                        spec, words[offset : offset + spec.word_count]
                    )
            decoded_by_id.update(decoded_here)

        for spec in specs:
            decoded = decoded_by_id.get(spec.id)
            if decoded is not None:
                readings.extend(self._readings_for(spec, decoded, now))
        readings.extend(self._derived_readings(specs, decoded_by_id, now))

        if readings:
            self._last_reading_at = time.time()
            self._error_count = 0
        return readings

    async def _pace(self) -> None:
        """Wait the configured gap between requests (cancellable)."""
        if self._request_spacing_s > 0:
            await asyncio.sleep(self._request_spacing_s)

    async def _read_block(
        self, unit_id: int, function_code: int, start: int, count: int
    ) -> list[int] | None:
        """One batched read. None on any failure, which triggers the fallback."""
        assert self._client is not None
        base = self._map.address_base if self._map else 0
        address = start - base
        if address < 0 or count < 1 or count > 125 or address + count > 65536:
            logger.warning(
                "vendor_map_block_out_of_range",
                start=start,
                count=count,
                unit_id=unit_id,
            )
            return None
        try:
            unit = {DEVICE_KWARG: unit_id}
            if function_code == 4:
                result = await self._client.read_input_registers(
                    address=address, count=count, **unit
                )
            else:
                result = await self._client.read_holding_registers(
                    address=address, count=count, **unit
                )
        except (ModbusIOException, OSError, asyncio.TimeoutError):
            self._error_count += 1
            logger.warning(
                "vendor_map_block_read_error", start=start, count=count, unit_id=unit_id
            )
            return None
        if result.isError():
            return None
        return [int(word) for word in result.registers]

    async def _read_block_individually(
        self, specs: list[RegisterSpec]
    ) -> dict[str, Decoded]:
        """Per-register fallback so one bad address cannot cost its neighbours."""
        decoded: dict[str, Decoded] = {}
        for index, spec in enumerate(specs):
            if index:
                await self._pace()
            words = await self._read_words(spec, spec.word_count)
            if words is None:
                continue
            decoded[spec.id] = decode_register(spec, words)
        return decoded

    def _readings_for(
        self, spec: RegisterSpec, decoded: Decoded, now: datetime
    ) -> list[Reading]:
        """Turn one decoded register into zero, one or two readings."""
        if decoded.refusal is not None:
            # A refusal is logged, never emitted as a value. Debug level: a 2G
            # site with a half-implemented firmware would otherwise fill the
            # log with the same line every poll.
            logger.debug(
                "vendor_map_no_reading",
                register=spec.id,
                metric=spec.metric,
                reason=decoded.refusal,
                raw=[f"0x{word:04X}" for word in decoded.raw_words],
            )
            return []

        raw_register = f"modbus:{spec.unit_id}:{spec.address}"
        if spec.data_type == "fault_bits" and decoded.raw_words:
            # The fault-code table is what a bench session builds, so the raw
            # words travel with the reading rather than being thrown away.
            raw_register += ":" + ",".join(f"{word:04X}" for word in decoded.raw_words)

        readings: list[Reading] = []
        value = primary_value(spec, decoded)
        if value is not None:
            readings.append(
                Reading(
                    metric_name=spec.metric,
                    value=value,
                    timestamp=now,
                    quality=decoded.quality,
                    raw_register=raw_register,
                    asset_id=spec.asset_id,
                )
            )
        split = split_signed(spec, decoded)
        if split is not None:
            readings.append(
                Reading(
                    metric_name=split[0],
                    value=split[1],
                    timestamp=now,
                    quality=decoded.quality,
                    raw_register=raw_register,
                    asset_id=spec.asset_id,
                )
            )
        return readings

    def _derived_readings(
        self,
        specs: list[RegisterSpec],
        decoded_by_id: dict[str, Decoded],
        now: datetime,
    ) -> list[Reading]:
        readings: list[Reading] = []
        for spec in specs:
            if spec.derived != "sum":
                continue
            result = derive_sum(spec, decoded_by_id)
            if result.value is None:
                logger.debug(
                    "vendor_map_derived_incomplete",
                    register=spec.id,
                    metric=spec.metric,
                    reason=result.refusal,
                )
                continue
            readings.append(
                Reading(
                    metric_name=spec.metric,
                    value=result.value,
                    timestamp=now,
                    quality=result.quality,
                    raw_register=f"derived:{'+'.join(spec.components)}",
                    asset_id=spec.asset_id,
                )
            )
        return readings

    async def subscribe(
        self,
        callback: Callable[[list[Reading]], Awaitable[None]],
    ) -> bool:
        """Modbus is poll-based; wrap poll in a timed loop."""
        interval = self._config.get("poll_interval_seconds", 60)
        while True:
            readings = await self.poll()
            if readings:
                await callback(readings)
            await asyncio.sleep(interval)

    async def write(self, metric_name: str, value: float) -> bool:
        """Write an installer-declared holding register, never an inferred one."""
        if not self._client or not self._client.connected:
            return False
        if self._map is not None:
            # Unreachable via config (initialize refuses the combination), but
            # a write path that depends on a validation elsewhere is one
            # refactor away from being a write path.
            logger.warning("vendor_map_write_refused", metric=metric_name)
            return False
        raw_mappings = self._config.get("write_mappings") or {}
        mapping = (
            raw_mappings.get(metric_name) if isinstance(raw_mappings, dict) else None
        )
        if not isinstance(mapping, dict):
            return False
        scale = float(mapping.get("scale", 1.0))
        if scale == 0:
            return False
        data_type = str(mapping.get("type", "uint16"))
        numeric = value / scale
        if data_type == "float32":
            registers = list(struct.unpack(">HH", struct.pack(">f", float(numeric))))
        else:
            raw = int(round(numeric))
            bits = 32 if data_type in {"uint32", "int32"} else 16
            if data_type.startswith("int") and raw < 0:
                raw += 1 << bits
            if not 0 <= raw < (1 << bits):
                return False
            registers = [raw] if bits == 16 else [(raw >> 16) & 0xFFFF, raw & 0xFFFF]
        unit_id = int(mapping.get("unit_id", self._config.get("unit_id", 1)))
        address = int(mapping["address"])
        if len(registers) == 1:
            result = await self._client.write_register(
                address=address, value=registers[0], **{DEVICE_KWARG: unit_id}
            )
        else:
            result = await self._client.write_registers(
                address=address, values=registers, **{DEVICE_KWARG: unit_id}
            )
        return not result.isError()

    async def health_check(self) -> AdapterStatus:
        """Return ERROR if no readings in 5 minutes or client disconnected."""
        if not self._client or not self._client.connected:
            return AdapterStatus.DISCONNECTED
        if self._last_reading_at > 0 and time.time() - self._last_reading_at > 300:
            return AdapterStatus.ERROR
        return AdapterStatus.CONNECTED

    def capabilities(self) -> AdapterCapabilities:
        if self._map is not None:
            metrics = sorted(
                {spec.metric for spec in self._specs}
                | {
                    spec.split_negative_to
                    for spec in self._specs
                    if spec.split_negative_to is not None
                }
            )
            # The shortest group cadence is the loop's tick; each group then
            # decides for itself whether it is due.
            interval = min(
                (group.poll_interval_seconds for group in self._map.groups),
                default=60,
            )
            return AdapterCapabilities(
                can_read=True,
                can_write=False,
                can_subscribe=False,
                supported_metrics=metrics,
                supported_write_metrics=[],
                poll_interval_seconds=interval,
                min_poll_interval_seconds=5,
                supports_asset_targeting=True,
            )
        return AdapterCapabilities(
            can_read=True,
            can_write=bool(self._config.get("write_mappings")),
            can_subscribe=False,
            supported_metrics=[r["metric"] for r in self._registers],
            supported_write_metrics=sorted(
                (self._config.get("write_mappings") or {}).keys()
            ),
            poll_interval_seconds=self._config.get("poll_interval_seconds", 60),
            min_poll_interval_seconds=5,
        )

    async def shutdown(self) -> None:
        """Close the Modbus client."""
        if self._client:
            self._client.close()
        self._status = AdapterStatus.DISCONNECTED
        logger.info("generic_modbus_shutdown", asset_id=self._asset_id)
