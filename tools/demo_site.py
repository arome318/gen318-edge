"""Run a simulated mini-grid day through the SMS service, no hardware needed.

    python tools/demo_site.py

Shows the SMS a technician and a community contact would receive as a site
goes from normal, to a falling battery, to a power cut, and back.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "edge"))

from sms.service import Contact, SmsConfig, SmsService  # noqa: E402
from sms.store import SmsStore  # noqa: E402

TECH, VILLAGE, NEIGHBOUR = "+2348030000001", "+2348030000002", "+2348030000009"


class PrintBackend:
    def send(self, phone: str, text: str) -> bool:
        who = {TECH: "technician", VILLAGE: "community"}.get(phone, "neighbour")
        print(f"   SMS -> {who:<10} | {text}")
        return True

    def receive(self) -> list[tuple[str, str]]:
        return []


def main() -> None:
    cfg = SmsConfig(site_name="Kuje", technicians=[Contact(TECH, "en")],
                    community=[Contact(VILLAGE, "pcm")], quiet_hours=None,
                    daily_summary_hour=None, hold_s=120)
    svc = SmsService(cfg, PrintBackend(), SmsStore(tempfile.mktemp(suffix=".db")))
    t = 1_763_050_000.0  # 17:06 WAT

    def step(minutes: float, **metrics: float) -> None:
        nonlocal t
        t += minutes * 60
        for name, value in metrics.items():
            svc.on_telemetry({"timestamp": t * 1000, "asset_id": "inv1",
                              "metric_name": name.replace("__", "."), "value": value})
        svc.tick(t)

    def ask(phone: str, text: str) -> None:
        reply = svc.handle_inbound(phone, text, t)
        print(f"   {phone[-4:]} texts {text!r:<10} -> {reply}")

    print("\n1. Normal evening")
    step(0, battery__soc_pct=62, solar__pv_power_w=1400, inverter__ac_output_voltage_v=230)
    ask(NEIGHBOUR, "POWER")
    print("\n2. Battery falls under 20% and stays there")
    step(60, battery__soc_pct=18, solar__pv_power_w=0, inverter__ac_output_voltage_v=230)
    step(3, battery__soc_pct=17, inverter__ac_output_voltage_v=230)
    print("\n3. Battery runs out: power cut")
    step(90, battery__soc_pct=4, inverter__ac_output_voltage_v=0)
    step(3, battery__soc_pct=4, inverter__ac_output_voltage_v=0)
    ask(NEIGHBOUR, "LIGHT")
    print("\n4. Technician confirms the alert was real")
    ask(TECH, "1 genset was off")
    print("\n5. Genset started, power back")
    step(20, battery__soc_pct=12, inverter__ac_output_voltage_v=231)
    print("\nLabelled data kept on the box (no phone numbers):")
    for row in svc.store.export_labelled():
        print("  ", row)


if __name__ == "__main__":
    main()
