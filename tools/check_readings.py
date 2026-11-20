"""Print the box's latest readings next to blanks for the inverter's own screen.

    python tools/check_readings.py            # every 10 seconds, Ctrl-C to stop

Write down what the inverter's display shows beside each line. Five matching
values (battery %, battery V, solar W, load W, output V) are what a bench
validation note needs.
"""

import json
import time
import urllib.request

KEY = [
    ("battery.soc_pct", "Battery %"), ("battery.voltage_v", "Battery V"),
    ("battery.power_w", "Battery W (+ charging)"), ("solar.pv_power_w", "Solar W"),
    ("solar.mppt1_power_w", "Solar W (PV1)"), ("load.power_w", "Load W"),
    ("inverter.ac_output_power_w", "Output W"), ("inverter.ac_output_voltage_v", "Output V"),
    ("grid.voltage_v", "Grid V"), ("inverter.temperature_c", "Inverter C"),
    ("inverter.error_code", "Fault code"),
]

while True:
    try:
        rows = json.load(urllib.request.urlopen("http://localhost:8080/metrics/latest", timeout=5))
    except Exception as exc:  # noqa: BLE001
        print("Box not answering on http://localhost:8080 :", exc)
        time.sleep(10)
        continue
    latest = {r["metric_name"]: r for r in rows}
    print("\n" + time.strftime("%H:%M:%S"), "  box value          inverter screen")
    for metric, label in KEY:
        if metric in latest:
            print(f"  {label:<24}{latest[metric]['value']:>10.1f}        ________")
    time.sleep(10)
