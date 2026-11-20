"""The local page's HTML (what a technician sees at http://<box>:8080).

Look and layout only: every figure here is handed in by ``api/main.py`` from
the same sources the page always used. The page is rendered on the box and
works with no internet: styles, script, fonts and logo are served from
``api/static``. ``static/page.js`` re-fetches the content (``?fragment=1``)
and swaps in what changed.

The Gen318 name and logo are trademarks (see NOTICE); the lockup below is
the brand's own artwork with its colours moved into CSS so the light theme
can re-ink it.
"""

from __future__ import annotations

import html
import time
from typing import Any

LABEL_NAMES = {1: "real", 2: "false alarm", 3: "other cause"}
INBOUND_NAMES = {"status": "asked for power status", "label": "marked an alert",
                 "lang": "changed language", "reset": "reset alert sensitivity",
                 "help": "asked for help", "unknown": "sent a message the box did not understand"}
# What an inbound SMS is shown as. The words people send are not stored, only
# the kind of message, so the bubble shows the command the box understood.
INBOUND_WORDS = {"status": "POWER", "lang": "LANG", "reset": "RESET TUNING", "help": "HELP", "unknown": "?"}

SOLAR_TOTAL = "solar.pv_power_w"
SOLAR_PARTS = tuple(f"solar.mppt{i}_power_w" for i in range(1, 5))
LOAD_METRICS = ("system.ac_consumption_w", "load.power_w", "inverter.ac_output_power_w")

LOCKUP = (
    '<svg class="lockup" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 255.4 64" role="img" aria-label="Gen318">'
    '<path class="lg-line" d="M32 32 V12 M32 32 L17.86 46.14 M32 32 L46.14 46.14" stroke-width="3.5"/>'
    '<circle class="lg-line" cx="32" cy="32" r="20" stroke-width="4"/>'
    '<rect class="lg-ink" x="29" y="9" width="6" height="6"/><rect class="lg-ink" x="43.14" y="14.86" width="6" height="6"/>'
    '<rect class="lg-ink" x="49" y="29" width="6" height="6"/><rect class="lg-ink" x="43.14" y="43.14" width="6" height="6"/>'
    '<rect class="lg-ink" x="29" y="49" width="6" height="6"/><rect class="lg-ink" x="14.86" y="43.14" width="6" height="6"/>'
    '<rect class="lg-ink" x="9" y="29" width="6" height="6"/><rect class="lg-ink" x="14.86" y="14.86" width="6" height="6"/>'
    '<rect class="lg-core" x="25.5" y="25.5" width="13" height="13" rx="2"/>'
    '<g transform="translate(80.8 19.5) scale(0.03505)"><path class="lg-ink" d="M416.0 711.0Q208.0 711.0 104.0 623.0Q0.0 535.0 0.0 356.0Q0.0 239.0 53.0 159.5Q106.0 80.0 206.5 40.0Q307.0 0.0 447.0 0.0Q538.0 0.0 613.0 15.5Q688.0 31.0 742.5 62.5Q797.0 94.0 826.5 142.0Q856.0 190.0 856.0 255.0L708.0 255.0Q708.0 219.0 687.0 194.0Q666.0 169.0 630.0 154.5Q594.0 140.0 548.5 133.5Q503.0 127.0 453.0 127.0Q389.0 127.0 334.5 139.5Q280.0 152.0 240.0 178.5Q200.0 205.0 178.0 246.0Q156.0 287.0 156.0 343.0L156.0 367.0Q156.0 445.0 190.0 492.5Q224.0 540.0 286.5 562.0Q349.0 584.0 434.0 584.0Q528.0 584.0 588.5 567.0Q649.0 550.0 678.5 518.0Q708.0 486.0 708.0 443.0L708.0 435.0L430.0 435.0L430.0 315.0L857.0 315.0L857.0 699.0L763.0 699.0L747.0 609.0Q710.0 645.0 658.5 667.5Q607.0 690.0 545.5 700.5Q484.0 711.0 416.0 711.0ZM1053.0 699.0L1053.0 12.0L1748.0 12.0L1748.0 139.0L1205.0 139.0L1205.0 286.0L1688.0 286.0L1688.0 413.0L1205.0 413.0L1205.0 572.0L1756.0 572.0L1756.0 699.0ZM1929.0 699.0L1929.0 12.0L2072.0 12.0L2448.0 373.0Q2461.0 384.0 2479.0 403.0Q2497.0 422.0 2516.5 442.0Q2536.0 462.0 2550.0 478.0L2558.0 478.0Q2557.0 456.0 2555.5 424.5Q2554.0 393.0 2554.0 373.0L2554.0 12.0L2700.0 12.0L2700.0 699.0L2560.0 699.0L2183.0 334.0Q2156.0 308.0 2126.5 276.5Q2097.0 245.0 2079.0 226.0L2072.0 226.0Q2073.0 239.0 2074.0 271.5Q2075.0 304.0 2075.0 346.0L2075.0 699.0ZM3188.0 711.0Q3076.0 711.0 3001.5 685.5Q2927.0 660.0 2890.0 614.5Q2853.0 569.0 2853.0 511.0L2853.0 495.0L2991.0 495.0L2991.0 509.0Q2991.0 554.0 3038.0 579.0Q3085.0 604.0 3171.0 604.0Q3269.0 604.0 3320.0 575.5Q3371.0 547.0 3371.0 494.0Q3371.0 457.0 3348.5 437.0Q3326.0 417.0 3288.5 409.5Q3251.0 402.0 3207.0 402.0L3093.0 402.0L3093.0 299.0L3199.0 299.0Q3237.0 299.0 3271.5 290.0Q3306.0 281.0 3328.5 259.5Q3351.0 238.0 3351.0 203.0Q3351.0 173.0 3329.5 151.5Q3308.0 130.0 3270.5 118.5Q3233.0 107.0 3185.0 107.0Q3138.0 107.0 3098.5 118.5Q3059.0 130.0 3035.0 150.5Q3011.0 171.0 3011.0 199.0L3011.0 212.0L2877.0 212.0L2877.0 191.0Q2877.0 137.0 2915.5 94.0Q2954.0 51.0 3025.5 25.5Q3097.0 0.0 3195.0 0.0Q3286.0 0.0 3352.5 23.5Q3419.0 47.0 3455.5 88.5Q3492.0 130.0 3492.0 186.0Q3492.0 228.0 3473.5 259.0Q3455.0 290.0 3424.5 310.0Q3394.0 330.0 3358.0 340.0L3358.0 344.0Q3429.0 357.0 3471.5 395.0Q3514.0 433.0 3514.0 513.0Q3514.0 571.0 3475.5 616.0Q3437.0 661.0 3364.5 686.0Q3292.0 711.0 3188.0 711.0ZM3653.0 699.0L3653.0 579.0L3877.0 579.0L3877.0 192.0L3653.0 192.0L3653.0 108.0Q3710.0 104.0 3761.5 93.0Q3813.0 82.0 3861.0 61.5Q3909.0 41.0 3953.0 10.0L4015.0 10.0L4015.0 579.0L4223.0 579.0L4223.0 699.0ZM4645.0 711.0Q4535.0 711.0 4459.5 687.5Q4384.0 664.0 4346.0 620.0Q4308.0 576.0 4308.0 515.0Q4308.0 454.0 4346.0 412.5Q4384.0 371.0 4451.0 348.0Q4390.0 317.0 4363.0 279.0Q4336.0 241.0 4336.0 193.0Q4336.0 135.0 4372.5 91.5Q4409.0 48.0 4478.0 24.0Q4547.0 0.0 4645.0 0.0Q4743.0 0.0 4812.0 24.0Q4881.0 48.0 4917.5 91.5Q4954.0 135.0 4954.0 193.0Q4954.0 241.0 4927.0 279.0Q4900.0 317.0 4839.0 348.0Q4907.0 371.0 4944.5 412.5Q4982.0 454.0 4982.0 515.0Q4982.0 576.0 4944.0 620.0Q4906.0 664.0 4831.0 687.5Q4756.0 711.0 4645.0 711.0ZM4645.0 604.0Q4740.0 604.0 4789.0 577.5Q4838.0 551.0 4838.0 505.0Q4838.0 455.0 4789.0 428.0Q4740.0 401.0 4645.0 401.0Q4550.0 401.0 4501.0 428.0Q4452.0 455.0 4452.0 505.0Q4452.0 551.0 4501.5 577.5Q4551.0 604.0 4645.0 604.0ZM4645.0 296.0Q4729.0 296.0 4778.0 271.0Q4827.0 246.0 4827.0 198.0Q4827.0 154.0 4779.5 130.5Q4732.0 107.0 4645.0 107.0Q4558.0 107.0 4510.5 130.5Q4463.0 154.0 4463.0 198.0Q4463.0 246.0 4512.0 271.0Q4561.0 296.0 4645.0 296.0Z"/></g>'
    "</svg>"
)


def esc(value: Any) -> str:
    return html.escape(str(value))


def _num(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}f}"


class Clock:
    """Formats times in the site's local time (the box keeps UTC)."""

    def __init__(self, utc_offset_hours: float) -> None:
        self.offset_s = utc_offset_hours * 3600
        whole = float(utc_offset_hours).is_integer()
        self.label = "WAT" if utc_offset_hours == 1 else f"UTC{utc_offset_hours:+g}" if whole else f"UTC{utc_offset_hours:+.1f}"

    def hms(self, ts: float) -> str:
        return time.strftime("%H:%M:%S", time.gmtime(ts + self.offset_s))

    def day_hm(self, ts: float) -> str:
        return time.strftime("%d %b %H:%M", time.gmtime(ts + self.offset_s))


# ── figures for the tiles (from the latest readings, as the old table showed them) ──


def _first(readings: list[dict[str, Any]], *metrics: str) -> float | None:
    for metric in metrics:
        vals = [float(r["value"]) for r in readings if r.get("metric_name") == metric]
        if vals:
            return sum(vals)
    return None


def _solar_w(readings: list[dict[str, Any]]) -> float | None:
    total = _first(readings, SOLAR_TOTAL)
    if total is not None:
        return total
    parts = [float(r["value"]) for r in readings if r.get("metric_name") in SOLAR_PARTS]
    return sum(parts) if parts else None


def comms_state(readings: list[dict[str, Any]], now: float, comms_lost_s: float) -> dict[str, Any]:
    """How fresh the readings are. Stale data must never look as sure as live data."""
    if not readings:
        return {"state": "waiting", "stale": False, "age_s": None, "last_ts": None}
    last_ts = max(float(r.get("timestamp", 0)) for r in readings) / 1000.0
    age = max(0.0, now - last_ts)
    lost = age > comms_lost_s
    # Dim well before the comms-lost alert on a real site (threshold 10 minutes):
    # three missed 30-second polls is already old data.
    stale = age > min(comms_lost_s, 90.0)
    return {"state": "lost" if lost else "ok", "stale": stale, "age_s": age, "last_ts": last_ts}


def _kpis(readings: list[dict[str, Any]], comms: dict[str, Any], clock: Clock, outage_v: float) -> str:
    dash = '<span class="muted">&ndash;</span>'
    asof = (f'<span class="chip small warning asof">As of {esc(clock.hms(comms["last_ts"]))}</span>'
            if comms.get("last_ts") else "")

    soc, volts = _first(readings, "battery.soc_pct"), _first(readings, "battery.voltage_v")
    battery_value = f"{_num(soc)}<small>%</small>" if soc is not None else dash
    battery_sub = f"{_num(volts, 1)} V" if volts is not None else "&nbsp;"

    pv = _solar_w(readings)
    solar_value = f"{_num(pv / 1000, 2)}<small>kW</small>" if pv is not None else dash
    pv_v = _first(readings, "solar.mppt1_voltage_v")
    solar_sub = f"{_num(pv_v)} V array" if pv_v is not None else "&nbsp;"

    load = _first(readings, *LOAD_METRICS)
    load_value = f"{_num(load / 1000, 2)}<small>kW</small>" if load is not None else dash
    amps = _first(readings, "inverter.ac_output_current_a")
    load_sub = f"{_num(amps, 1)} A out" if amps is not None else "&nbsp;"

    code, out_v = _first(readings, "inverter.error_code"), _first(readings, "inverter.ac_output_voltage_v")
    temp = _first(readings, "inverter.temperature_c")
    bad = False
    if code is not None and int(code) != 0:
        inverter_value, bad = f"FAULT<small>{int(code)}</small>", True
    elif out_v is not None and out_v < outage_v:
        inverter_value, bad = "OFF", True
    elif out_v is not None or code is not None:
        inverter_value = "ON"
    else:
        inverter_value = dash
    bits = ([f"{_num(out_v)} V"] if out_v is not None else []) + ([f"{_num(temp)} &deg;C"] if temp is not None else [])
    inverter_sub = " &middot; ".join(bits) or "&nbsp;"

    def tile(kind: str, label: str, value: str, sub: str, extra: str = "") -> str:
        return (f'<div class="card kpi {kind}{extra}"><div class="label">{label}</div>{asof}'
                f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')

    return (tile("battery", "Battery", battery_value, battery_sub)
            + tile("solar", "Solar", solar_value, solar_sub)
            + tile("load", "Load", load_value, load_sub)
            + tile("inverter", "Inverter", inverter_value, inverter_sub, " bad" if bad else ""))


# ── SMS feed ──


def _feed(messages: list[dict[str, Any]], labels: list[dict[str, Any]], clock: Clock) -> str:
    """Oldest first, like a phone: the box's messages on the right, incoming on the left."""
    label_at = {round(float(item["at"]), 3): item for item in labels}
    out = []
    for m in sorted(messages, key=lambda item: (item["at"], item["direction"] == "out")):
        when = esc(clock.hms(m["at"]))
        ident = esc(f'{m["direction"]}-{m["at"]:.3f}-{m["kind"]}-{m["role"]}')
        if m["direction"] == "out":
            failed = "" if m["ok"] else ' <span class="failed">Not sent</span>'
            out.append(
                f'<div class="msg out" data-id="{ident}"><div class="who">Box &rarr; {esc(m["role"])} &middot; {when}</div>'
                f'<div class="bubble">{esc(m["text"] or "(" + m["kind"] + ")")}</div>'
                f'{f"<div class=meta>{failed}</div>" if failed else ""}</div>')
            continue
        label = label_at.get(round(float(m["at"]), 3))
        if m["kind"] == "label" and label:
            word, who = str(label["label"]), "Technician"
        else:
            word, who = INBOUND_WORDS.get(m["kind"], "?"), "Phone"
        out.append(
            f'<div class="msg in" data-id="{ident}"><div class="who">{who} &rarr; box &middot; {when}</div>'
            f'<div class="bubble">{esc(word)}</div>'
            f'<div class="meta">Someone {esc(INBOUND_NAMES.get(m["kind"], m["kind"]))}</div></div>')
    return "".join(out) or '<p class="empty">No SMS yet.</p>'


# ── alerts ──


def _alerts(alerts: list[dict[str, Any]], clock: Clock) -> str:
    rows = []
    for a in alerts:
        level = a["severity"] if a["severity"] in ("critical", "warning") else "info"
        is_open = a["cleared_at"] is None
        marked = LABEL_NAMES.get(a["label"]) if a["label"] else None
        when = f'#{esc(a["ref"])} &middot; raised {esc(clock.day_hm(a["raised_at"]))}'
        if not is_open:
            when += f' &middot; cleared {esc(clock.hms(a["cleared_at"]))}'
        tags = f'<span class="badge">{"open" if is_open else "cleared"}</span>'
        if marked:
            tags += f'<span class="badge">marked {esc(marked)}</span>'
        rows.append(
            f'<div class="arow {level} {"open" if is_open else "cleared"}">'
            f'<span class="badge {level}">{esc(level)}</span>'
            f'<span class="name">{esc(a["code"].replace("_", " "))}</span>'
            f'<span class="tags">{tags}</span><span class="when">{when}</span></div>')
    return "".join(rows) or '<p class="empty">No alerts yet.</p>'


# ── smaller cards ──


def _learned(meta: dict[str, Any], clock: Clock) -> str:
    rows = "".join(
        f"<tr><td>{esc(k.split(':')[0])} <span class=\"muted mono\">{esc(k.split(':')[1])[:8]}</span></td>"
        f'<td class="n">{esc(clock.day_hm(v["trained_at"]))}</td><td class="n">{esc(v.get("days", "-"))}</td></tr>'
        for k, v in sorted(meta.items())
        if isinstance(v, dict) and k.count(":") == 1 and "trained_at" in v
    )
    if not rows:
        return '<p class="empty">Still collecting data.</p>'
    return ('<table><tr><th>Model</th><th class="n">Trained</th><th class="n">Days of data</th></tr>'
            f"{rows}</table>")


def _tuning(tuning: list[dict[str, Any]]) -> str:
    rows = "".join(
        f"<tr><td>{esc(t['code'].replace('_', ' '))}</td><td class=\"n\">{esc(t['steps'])} of {esc(t['max_steps'])}</td>"
        f"<td class=\"n\">{esc(t.get('was', '-'))}</td><td class=\"n\">{esc(t.get('now', 'set by the model'))}</td></tr>"
        for t in tuning
    )
    if not rows:
        return '<p class="empty">Nothing changed: all warnings are at the installer settings.</p>'
    return ('<table><tr><th>Warning</th><th class="n">Step</th><th class="n">Installer set</th>'
            f'<th class="n">Now</th></tr>{rows}</table>')


def _energy(en: dict[str, Any]) -> str:
    if en.get("available"):
        share = (f"About {esc(en['share_of_site_pct'])}% of what the site used in the last day."
                 if "share_of_site_pct" in en else "")
        box = (f'<p class="stat">About {esc(en["average_watts"])} W, {esc(en["wh_per_day"])} Wh a day '
               f'({esc(en["method"])})</p><p class="note">{share}</p>')
    else:
        box = '<p class="empty">Not enough samples yet (needs about an hour).</p>'
    models = en.get("models") or {"count": 0, "bytes": 0, "last_training_seconds": 0}
    if models["count"]:
        box += (f'<p class="note"><span class="mono">{esc(models["count"])}</span> model files, '
                f'<span class="mono">{esc(round(models["bytes"] / 1024))} kB</span> in all; the last training took '
                f'<span class="mono">{esc(models["last_training_seconds"])} s</span> on this box.</p>')
    else:
        box += '<p class="note">No models trained yet.</p>'
    return box


def _readings(readings: list[dict[str, Any]]) -> str:
    rows = "".join(
        f"<tr><td>{esc(r.get('asset_id'))}</td><td>{esc(r.get('metric_name'))}</td>"
        f"<td class=\"n\">{esc(round(float(r.get('value', 0)), 2))}</td></tr>"
        for r in readings[:200]
    )
    if not rows:
        return '<p class="empty">No readings yet.</p>'
    return ('<table class="readings"><tr><th>Asset</th><th>Metric</th><th class="n">Value</th></tr>'
            f"{rows}</table>")


def _comms_chip(comms: dict[str, Any]) -> str:
    if comms["state"] == "lost":
        minutes = max(1, round(comms["age_s"] / 60))
        return f'<span class="chip danger"><span class="glyph"></span>Comms lost &middot; {minutes} min</span>'
    if comms["state"] == "waiting":
        return '<span class="chip"><span class="glyph"></span>Waiting for readings</span>'
    return '<span class="chip ok"><span class="glyph"></span>Comms OK</span>'


def render_body(s: dict[str, Any]) -> str:
    """Everything that changes between refreshes. Each [data-live] block is swapped on its own."""
    clock = Clock(s["utc_offset_hours"])
    comms = comms_state(s["readings"], s["now"], s["comms_lost_s"])
    n_msgs = sum(1 for m in s["messages"] if m["direction"] == "out")
    open_alerts = sum(1 for a in s["alerts"] if a["cleared_at"] is None)
    return f"""<header class="top">
{LOCKUP}<span class="rule"></span>
<div class="site"><h1>{esc(s["site_name"])}</h1><span class="label">Edge box &middot; {esc(s["device_id"])}</span></div>
<div class="status"><span class="fresh"><span class="live-dot"></span><span id="ago">Updated just now</span></span>
<span id="comms" data-live>{_comms_chip(comms)}</span></div>
</header>
<section id="kpis" data-live class="kpis{' stale' if comms['stale'] else ''}">{_kpis(s["readings"], comms, clock, s["outage_voltage_v"])}</section>
<div class="grid2">
<section class="card"><div class="head"><h2>SMS</h2>
<span class="note" id="sms-count" data-live><span class="mono">{n_msgs}</span> sent &middot; times are {esc(clock.label)}</span></div>
<div id="feed" data-live class="feed">{_feed(s["messages"], s["labels"], clock)}</div>
<p class="foot">Sent over the mobile network, no internet. Phone numbers are never shown or stored.</p></section>
<section class="card"><div class="head"><h2>Alerts</h2>
<span class="note" id="alert-count" data-live><span class="mono">{open_alerts}</span> open</span></div>
<div id="alerts" data-live class="rows">{_alerts(s["alerts"], clock)}</div>
<p class="foot">To mark an alert, reply to its SMS: 1 = real, 2 = false alarm, 3 = other cause.</p></section>
</div>
<div class="grid3">
<section class="card"><div class="head"><h2>What this box has learned</h2></div>
<div id="learned" data-live>{_learned(s["learned"], clock)}</div>
<p class="foot">Models train here from this site's own data: solar, load and battery run-out after 7 days, battery health after 14.</p></section>
<section class="card"><div class="head"><h2>Alert sensitivity changed by your replies</h2></div>
<div id="tuning" data-live>{_tuning(s["tuning"])}</div>
<p class="foot">Three false-alarm replies in the last five make a warning less sensitive (two steps at most).
A reply of 1 (real), or RESET TUNING, puts it back. Critical alerts never change.</p></section>
<section class="card"><div class="head"><h2>What this box itself uses</h2></div>
<div id="energy" data-live>{_energy(s["energy"])}</div></section>
</div>
<section class="card"><details><summary><h2>Latest readings</h2></summary>
<div id="readings" data-live>{_readings(s["readings"])}</div></details></section>
<p class="foot">Works offline. This page updates itself every {int(s["refresh"])} seconds.</p>"""


def render(s: dict[str, Any], fragment: bool = False) -> str:
    body = render_body(s)
    if fragment:
        return f"<!doctype html><html><body>{body}</body></html>"
    theme = f' data-theme="{s["theme"]}"' if s.get("theme") in ("dark", "light") else ""
    refresh = int(s["refresh"])
    return f"""<!doctype html>
<html lang="en"{theme} data-refresh="{refresh}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>{esc(s["site_name"])} &middot; Gen318 Edge</title>
<link rel="icon" href="static/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="static/page.css">
<noscript><meta http-equiv="refresh" content="{refresh}"></noscript>
</head><body><div class="wrap">
{body}
</div><script src="static/page.js" defer></script></body></html>"""
