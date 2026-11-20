"""Pretend to be a Felicity IVEM inverter on a serial port (for testing).

    socat -d -d pty,raw,echo=0,link=/tmp/inv pty,raw,echo=0,link=/tmp/box &
    python tools/felicity_emulator.py /tmp/inv

Then point the box's Felicity adapter at /tmp/box. Register addresses and
scales follow edge/collector/adapters/vendor_maps/felicity_hybrid.yaml; the
values are made up. It proves the box's software path end to end, NOT that
the map matches a real inverter (only a real unit can show that).
"""

from __future__ import annotations

import asyncio
import math
import sys
import time

from pymodbus.server import StartAsyncSerialServer
from pymodbus.simulator import DataType, SimData, SimDevice

BASE = 4355  # first register the map reads (0x1103)
SIZE = 4640 - BASE


def s16(v: int) -> int:
    return v & 0xFFFF


def snapshot(t: float) -> dict[int, int]:
    wobble = math.sin(t / 30)
    load_w = int(1800 + 300 * wobble)
    pv_w = int(1200 + 200 * math.cos(t / 45))
    return {
        4355: 0,                        # inverter fault code
        4360: int(5120 + 20 * wobble),  # battery V x0.01  -> 51.2 V
        4362: s16(pv_w - load_w),       # battery W, + = charging
        4366: 5000,                     # output Hz x0.01
        4369: 2300,                     # output V x0.1    -> 230 V
        4375: 0, 4377: 0,               # no grid connected
        4382: s16(load_w),              # load W
        4386: 42, 4387: 31,             # inverter / battery temperature C
        4390: 2400, 4393: 50,           # PV1 V x0.1, A x0.1
        4394: s16(pv_w),                # PV1 W
        4502: 0, 4503: int(load_w / 23), 4509: 0,
        4624: 640, 4625: 980,           # SoC / SoH x0.1
    }


async def main(port: str) -> None:
    async def action(_function_code: int, block_start: int, address: int, count: int,
                     registers: list[int], _written: list | None) -> None:
        """Refresh the requested registers with live made-up values."""
        values = snapshot(time.time())
        for i in range(count):
            registers[address - block_start + i] = values.get(address + i, 0)

    device = SimDevice(
        id=1,
        simdata=[SimData(address=BASE, count=SIZE, values=0, datatype=DataType.REGISTERS)],
        action=action,
    )
    print(f"Felicity emulator on {port} (9600 8N1, unit 1)", flush=True)
    await StartAsyncSerialServer(context=device, port=port, baudrate=9600)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/inv"))
