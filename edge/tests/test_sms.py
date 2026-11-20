"""Offline SMS service: rules, debouncing, templates, commands, modem parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sms.backends import ATModemBackend, DryRunBackend, parse_cmgl
from sms.commands import parse
from sms.rules import FaultTracker, SiteState, Thresholds, evaluate
from sms.service import Contact, SmsConfig, SmsService
from sms.store import SmsStore, normalise_phone
from sms.templates import Templates

T0 = 1_763_000_000.0  # fixed epoch for deterministic tests
TECH = "+2348030000001"
NEIGHBOUR = "+2348030000009"


class FakeBackend:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.inbox: list[tuple[str, str]] = []

    def send(self, phone: str, text: str) -> bool:
        self.sent.append((phone, text))
        return True

    def receive(self) -> list[tuple[str, str]]:
        out, self.inbox = self.inbox, []
        return out


def feed(svc: SmsService, metric: str, value: float, ts: float, asset: str = "inv1") -> None:
    svc.on_telemetry({"timestamp": ts * 1000, "asset_id": asset, "metric_name": metric, "value": value})


@pytest.fixture
def svc(tmp_path: Path) -> tuple[SmsService, FakeBackend]:
    cfg = SmsConfig(
        site_name="Kuje",
        technicians=[Contact(TECH, "en", "Musa")],
        community=[Contact("+2348030000002", "pcm")],
        quiet_hours=None,
        daily_summary_hour=None,
        hold_s=60,
    )
    backend = FakeBackend()
    return SmsService(cfg, backend, SmsStore(str(tmp_path / "sms.db"))), backend


# ── rules ──────────────────────────────────────────────────────


def test_rules_detect_each_fault() -> None:
    s = SiteState()
    for metric, value in {
        "battery.soc_pct": 8, "battery.temperature_c": 50,
        "battery.cell_max_voltage_v": 3.45, "battery.cell_min_voltage_v": 3.20,
        "inverter.error_code": 7, "inverter.temperature_c": 75,
        "inverter.ac_output_power_w": 4800, "inverter.ac_output_voltage_v": 0,
    }.items():
        s.update("inv1", metric, value, T0)
    codes = {f.code for f in evaluate(s, Thresholds(), T0, {"inv1": 5000})}
    assert codes == {"battery_critical", "battery_hot", "cell_imbalance", "inverter_fault",
                     "inverter_hot", "overload", "power_off"}


def test_rules_warning_and_diesel_waste() -> None:
    s = SiteState()
    s.update("a", "battery.soc_pct", 15, T0)
    s.update("g", "battery.soc_pct", 80, T0)
    s.update("g", "genset.power_w", 3000, T0)
    faults = {(f.asset_id, f.code) for f in evaluate(s, Thresholds(), T0)}
    assert ("a", "battery_low") in faults and ("g", "diesel_waste") in faults


def test_comms_lost_suppresses_other_rules() -> None:
    s = SiteState()
    s.update("inv1", "battery.soc_pct", 5, T0)
    faults = evaluate(s, Thresholds(), T0 + 700)
    assert [f.code for f in faults] == ["comms_lost"]


def test_stale_values_are_ignored() -> None:
    s = SiteState()
    s.update("inv1", "battery.soc_pct", 5, T0)
    s.update("inv1", "inverter.state", 1, T0 + 950)
    assert evaluate(s, Thresholds(), T0 + 950) == []


def test_tracker_holds_then_alerts_once_then_clears() -> None:
    s = SiteState()
    s.update("inv1", "battery.soc_pct", 5, T0)
    tr = FaultTracker(hold_s=60, realert_s=3600)
    f = evaluate(s, Thresholds(), T0)
    assert tr.step(f, T0) == ([], [])
    alerted, _ = tr.step(f, T0 + 61)
    assert [x.code for x in alerted] == ["battery_critical"]
    assert tr.step(f, T0 + 120) == ([], [])
    again, _ = tr.step(f, T0 + 61 + 3600)
    assert len(again) == 1
    _, cleared = tr.step([], T0 + 4000)
    assert [x.code for x in cleared] == ["battery_critical"]


def test_tracker_flapping_fault_never_alerts() -> None:
    s = SiteState()
    s.update("inv1", "battery.soc_pct", 5, T0)
    tr = FaultTracker(hold_s=60)
    f = evaluate(s, Thresholds(), T0)
    tr.step(f, T0)
    tr.step([], T0 + 30)
    assert tr.step(f, T0 + 61) == ([], [])


# ── templates and commands ─────────────────────────────────────


def test_every_fault_has_a_template_in_every_language() -> None:
    t = Templates()
    codes = ["battery_critical", "battery_low", "battery_hot", "cell_imbalance", "inverter_fault",
             "inverter_hot", "overload", "power_off", "diesel_waste", "comms_lost", "box_alert"]
    values = dict(soc=9, temp=50, spread_mv=200, error_code=7, load_pct=96, voltage=0,
                  minutes=12, detail="lid open")
    assert {"en", "pcm"} <= set(t.languages)
    for lang in t.languages:
        for code in codes:
            text = t.render(lang, "faults", code, **values)
            assert text.isascii() and len(text) <= 160, (lang, code)
            assert t.render(lang, "names", code)
            assert t.render(lang, "recovered", code)


@pytest.mark.parametrize("text,kind", [
    ("POWER", "status"), ("light?", "status"), (" wuta ", "status"), ("help", "help"),
    ("LANG pcm", "lang"), ("hello", "unknown"), ("", "unknown"),
])
def test_parse_kinds(text: str, kind: str) -> None:
    assert parse(text).kind == kind


def test_parse_labels() -> None:
    assert parse("1").label == 1
    c = parse("2 #ab3k fan was dusty")
    assert (c.kind, c.label, c.ref, c.note) == ("label", 2, "AB3K", "fan was dusty")
    assert parse("3 cable loose").note == "cable loose"


def test_normalise_phone() -> None:
    assert normalise_phone("0803 000 0001") == TECH
    assert normalise_phone("2348030000001") == TECH
    assert normalise_phone("+2348030000001") == TECH


# ── service ────────────────────────────────────────────────────


def test_alert_goes_to_technician_with_reply_code(svc) -> None:
    service, backend = svc
    feed(service, "battery.soc_pct", 8, T0)
    service.tick(T0)
    assert backend.sent == []
    feed(service, "battery.soc_pct", 8, T0 + 61)
    service.tick(T0 + 61)
    assert len(backend.sent) == 1
    phone, text = backend.sent[0]
    assert phone == TECH and text.startswith("Kuje: Battery very low (8%)") and "Reply 1=real" in text


def test_power_off_notifies_community_in_pidgin_and_restore(svc) -> None:
    service, backend = svc
    for t in (T0, T0 + 61):
        feed(service, "inverter.ac_output_voltage_v", 0, t)
        service.tick(t)
    community = [m for m in backend.sent if m[0] == "+2348030000002"]
    assert community and "Light no dey" in community[0][1]
    feed(service, "inverter.ac_output_voltage_v", 230, T0 + 120)
    service.tick(T0 + 120)
    assert any("E don correct" in m[1] for m in backend.sent if m[0] == "+2348030000002")
    assert any("OK again: power is back on" in m[1] for m in backend.sent if m[0] == TECH)


def test_power_status_reply_for_anyone(svc) -> None:
    service, _ = svc
    feed(service, "battery.soc_pct", 64, T0)
    feed(service, "solar.pv_power_w", 3200, T0)
    service.tick(T0)
    reply = service.handle_inbound("08030000009", "power", T0 + 5)
    assert reply == "Kuje: Power ON. Battery 64%. Solar 3.2kW."


def test_status_unknown_without_data(svc) -> None:
    service, _ = svc
    assert "No recent data" in service.handle_inbound(NEIGHBOUR, "POWER", T0)


def test_status_during_outage_names_cause(svc) -> None:
    service, _ = svc
    feed(service, "inverter.ac_output_voltage_v", 0, T0)
    feed(service, "battery.soc_pct", 4, T0)
    service.tick(T0)
    reply = service.handle_inbound(NEIGHBOUR, "LIGHT", T0 + 30)
    assert "Power OFF since" in reply and "battery very low" in reply


def test_status_rate_limit(svc) -> None:
    service, _ = svc
    replies = [service.handle_inbound(NEIGHBOUR, "POWER", T0 + i) for i in range(5)]
    assert all(replies[:3]) and replies[4] is None


def test_technician_label_attaches_to_latest_alert(svc) -> None:
    service, backend = svc
    for t in (T0, T0 + 61):
        feed(service, "inverter.error_code", 7, t)
        service.tick(t)
    ref = backend.sent[-1][1].rsplit("#", 1)[1]
    reply = service.handle_inbound("08030000001", "2 fan was dusty", T0 + 100)
    assert reply == f"Thanks. Alert #{ref} marked: false alarm."
    exported = service.store.export_labelled()
    assert exported == [{"code": "inverter_fault", "severity": "critical", "raised_at": T0 + 61,
                         "cleared_at": None, "values": {"error_code": 7}, "label": 2}]
    assert "phone" not in json.dumps(exported)


def test_labels_from_non_technicians_are_ignored(svc) -> None:
    service, _ = svc
    assert service.handle_inbound(NEIGHBOUR, "1", T0) is None


def test_language_switch(svc) -> None:
    service, _ = svc
    assert service.handle_inbound(NEIGHBOUR, "LANG PCM", T0) == "Language don change to Pidgin."
    assert "Box never get new data" in service.handle_inbound(NEIGHBOUR, "POWER", T0 + 1)


def test_unknown_message_gets_help_but_rate_limited(svc) -> None:
    service, _ = svc
    replies = [service.handle_inbound(NEIGHBOUR, "hi", T0 + i) for i in range(4)]
    assert replies[0].startswith("Send POWER") and replies[3] is None


def test_quiet_hours_defer_warnings_but_not_critical(tmp_path: Path) -> None:
    cfg = SmsConfig(site_name="S", technicians=[Contact(TECH)], quiet_hours=(22, 6),
                    daily_summary_hour=None, hold_s=0)
    backend = FakeBackend()
    service = SmsService(cfg, backend, SmsStore(str(tmp_path / "q.db")))
    night = 1_763_074_800.0  # 23:00 WAT
    feed(service, "battery.soc_pct", 15, night)
    service.tick(night)
    assert backend.sent == []
    morning = night + 8 * 3600  # 07:00 WAT
    feed(service, "battery.soc_pct", 15, morning)
    service.tick(morning)
    assert len(backend.sent) == 1 and "Battery low" in backend.sent[0][1]


def test_daily_summary(tmp_path: Path) -> None:
    cfg = SmsConfig(site_name="S", technicians=[Contact(TECH)], quiet_hours=None,
                    daily_summary_hour=19, hold_s=0)
    backend = FakeBackend()
    service = SmsService(cfg, backend, SmsStore(str(tmp_path / "d.db")))
    seven_pm = 1_763_056_800.0  # 19:00 WAT
    feed(service, "battery.soc_pct", 55, seven_pm)
    service.tick(seven_pm)
    service.tick(seven_pm + 60)
    summaries = [m for m in backend.sent if "today" in m[1]]
    assert len(summaries) == 1 and "power on 100%" in summaries[0][1] and "lowest battery 55%" in summaries[0][1]


def test_box_alert_cooldown(svc) -> None:
    service, backend = svc
    service.on_box_alert({"type": "tamper", "message": "lid opened"}, now=T0)
    service.on_box_alert({"type": "tamper", "message": "lid opened"}, now=T0 + 60)
    assert len(backend.sent) == 1 and "lid opened" in backend.sent[0][1]


def test_config_from_yaml() -> None:
    cfg = SmsConfig.from_yaml(
        "site_name: Kuje\ntechnicians:\n  - {name: Musa, phone: '0803 000 0001', lang: pcm}\n"
        "thresholds: {soc_warning_pct: 25, unknown_key: 1}\nrated_power_w: {inv1: 5000}\n"
        "backend: {type: at_modem, port: /dev/ttyUSB2}\n"
    )
    assert cfg.technicians[0].phone == TECH and cfg.technicians[0].lang == "pcm"
    assert cfg.thresholds.soc_warning_pct == 25 and cfg.rated_power_w == {"inv1": 5000.0}


# ── backends ───────────────────────────────────────────────────


def test_dry_run_backend_round_trip(tmp_path: Path) -> None:
    b = DryRunBackend(str(tmp_path))
    assert b.send(TECH, "hello")
    (tmp_path / "inbox.jsonl").write_text(json.dumps({"from": TECH, "text": "POWER"}) + "\n")
    assert b.receive() == [(TECH, "POWER")]
    assert b.receive() == []


def test_parse_cmgl_listing() -> None:
    resp = ('\r\n+CMGL: 1,"REC UNREAD","+2348030000001",,"26/11/11,10:00:00+04"\r\nPOWER\r\n'
            '+CMGL: 4,"REC READ","+2348030000002",,"26/11/11,10:01:00+04"\r\n2 #AB3K\r\n\r\nOK\r\n')
    msgs, idx = parse_cmgl(resp)
    assert msgs == [("+2348030000001", "POWER"), ("+2348030000002", "2 #AB3K")]
    assert idx == [1, 4]


class FakeSerial:
    """Minimal modem: answers OK, prompts '>' for CMGS, lists one SMS."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self._pending = b""

    def write(self, data: bytes) -> None:
        self.written.append(data)
        if data.startswith(b"AT+CMGS"):
            self._pending += b"\r\n> "
        elif data.endswith(b"\x1a"):
            self._pending += b"\r\n+CMGS: 12\r\n\r\nOK\r\n"
        elif data.startswith(b'AT+CMGL'):
            self._pending += (b'\r\n+CMGL: 2,"REC UNREAD","+2348030000009",,"26/11/11,10:00:00+04"'
                              b"\r\nLIGHT\r\n\r\nOK\r\n")
        else:
            self._pending += b"\r\nOK\r\n"

    def read(self, n: int) -> bytes:
        out, self._pending = self._pending[:n], self._pending[n:]
        return out

    def close(self) -> None:
        pass


def test_at_modem_send_and_receive() -> None:
    fake = FakeSerial()
    modem = ATModemBackend("/dev/null", serial_factory=lambda: fake)
    assert modem.send(TECH, "Test alert")
    assert b'AT+CMGS="+2348030000001"\r' in fake.written
    assert b"Test alert\x1a" in fake.written
    assert modem.receive() == [("+2348030000009", "LIGHT")]
    assert b"AT+CMGD=2\r" in fake.written


def test_technician_page_and_sms_endpoint(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    import api.main as api_main
    from common.database import EdgeDatabase

    sms_db = tmp_path / "sms.db"
    store = SmsStore(str(sms_db))
    store.record_alert("AB3K", "inverter_fault", "critical", "inv1", {"error_code": 7}, T0)
    store.record_label("AB3K", 2, "dusty", TECH, T0 + 10)
    store.close()
    monkeypatch.setattr(api_main, "SMS_DB_PATH", str(sms_db))
    db = EdgeDatabase(str(tmp_path / "edge.db"), "")
    db.connect()
    db.init_schema()
    monkeypatch.setattr(api_main, "_db", db)
    client = TestClient(api_main.app)
    alerts = client.get("/sms/alerts").json()
    assert alerts[0]["ref"] == "AB3K" and alerts[0]["label"] == 2
    page = client.get("/").text
    assert "#AB3K" in page and "inverter fault" in page and "false alarm" in page


def test_status_and_overload_with_per_tracker_metrics(svc) -> None:
    service, _ = svc
    feed(service, "battery.soc_pct", 70, T0)
    feed(service, "solar.mppt1_power_w", 900, T0)
    feed(service, "solar.mppt2_power_w", 600, T0)
    assert service.handle_inbound(NEIGHBOUR, "POWER", T0 + 1) == "Kuje: Power ON. Battery 70%. Solar 1.5kW."
    s = SiteState()
    s.update("inv1", "load.power_w", 4900, T0)
    assert [f.code for f in evaluate(s, Thresholds(), T0, {"inv1": 5000})] == ["overload"]
