"""The SMS service: alerts out, status and labels in."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

import structlog
import yaml

from sms.backends import SmsBackend
from sms.commands import parse
from sms.rules import Fault, FaultTracker, SiteState, Thresholds, evaluate
from sms.store import SmsStore, normalise_phone
from sms.templates import Templates
from sms.tuning import LEARNED_CODES, MAX_STEPS, decide, tuned_thresholds

logger = structlog.get_logger()

DAY = 86400


@dataclass
class Contact:
    phone: str
    lang: str = "en"
    name: str = ""


@dataclass
class SmsConfig:
    site_name: str = "Site"
    default_lang: str = "en"
    utc_offset_hours: float = 1.0  # Nigeria (WAT), no daylight saving
    technicians: list[Contact] = field(default_factory=list)
    community: list[Contact] = field(default_factory=list)
    public_status: bool = True
    status_per_hour: int = 3
    help_per_day: int = 2
    notify_community_on: list[str] = field(default_factory=lambda: ["power_off"])
    quiet_hours: tuple[int, int] | None = (22, 6)
    # Warnings that are useless in the morning, so they do not wait out quiet hours.
    urgent_warnings: list[str] = field(default_factory=lambda: ["battery_runout"])
    # Let technician replies (1 real, 2 false alarm) adjust warning sensitivity.
    auto_tune: bool = True
    daily_summary_hour: int | None = 19
    hold_s: int = 120
    realert_s: int = 6 * 3600
    label_window_s: int = DAY
    thresholds: Thresholds = field(default_factory=Thresholds)
    rated_power_w: dict[str, float] = field(default_factory=dict)
    backend: dict[str, Any] = field(default_factory=lambda: {"type": "dry_run"})

    @classmethod
    def from_yaml(cls, text: str) -> SmsConfig:
        d = yaml.safe_load(text) or {}

        def contacts(items: Any) -> list[Contact]:
            return [Contact(phone=normalise_phone(str(c["phone"])), lang=c.get("lang", "en"),
                            name=c.get("name", "")) for c in (items or [])]

        qh = d.get("quiet_hours", [22, 6])
        return cls(
            site_name=d.get("site_name", "Site"),
            default_lang=d.get("default_lang", "en"),
            utc_offset_hours=float(d.get("utc_offset_hours", 1.0)),
            technicians=contacts(d.get("technicians")),
            community=contacts(d.get("community")),
            public_status=bool(d.get("public_status", True)),
            status_per_hour=int(d.get("status_per_hour", 3)),
            help_per_day=int(d.get("help_per_day", 2)),
            notify_community_on=list(d.get("notify_community_on", ["power_off"])),
            quiet_hours=tuple(qh) if qh else None,  # type: ignore[arg-type]
            urgent_warnings=list(d.get("urgent_warnings", ["battery_runout"]) or []),
            auto_tune=bool(d.get("auto_tune", True)),
            daily_summary_hour=d.get("daily_summary_hour", 19),
            hold_s=int(d.get("hold_s", 120)),
            realert_s=int(d.get("realert_s", 6 * 3600)),
            thresholds=Thresholds.from_dict(d.get("thresholds")),
            rated_power_w={str(k): float(v) for k, v in (d.get("rated_power_w") or {}).items()},
            backend=d.get("backend") or {"type": "dry_run"},
        )


class SmsService:
    def __init__(self, cfg: SmsConfig, backend: SmsBackend, store: SmsStore,
                 templates: Templates | None = None,
                 publish_tuning: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.cfg = cfg
        self._publish_tuning = publish_tuning
        self._tuning_published_at = 0.0
        self.backend = backend
        self.store = store
        self.t = templates or Templates()
        self.state = SiteState()
        self.tracker = FaultTracker(cfg.hold_s, cfg.realert_s)
        self._tech_phones = {c.phone for c in cfg.technicians}
        self._deferred: list[tuple[Fault, str]] = []
        self._refs: dict[str, str] = {}
        self._outage_since: float | None = None
        self._outage_seconds_today = 0.0
        self._min_soc_today: float | None = None
        self._last_tick: float | None = None
        self._day: int | None = None
        self._summary_sent_day: int | None = None
        self._box_alert_last: dict[str, float] = {}
        self._learned: list[Fault] = []
        self._learned_at: float | None = None
        self.tuning: dict[str, int] = {}
        self.thresholds = cfg.thresholds
        self._retune(announce=False)

    # ── sensitivity from technician replies ─────────────────────
    def _retune(self, announce: bool, now: float | None = None) -> None:
        """Recompute steps from the stored labels; tell technicians what changed."""
        if not self.cfg.auto_tune:
            return
        since = float(self.store.get_meta("tuning_reset_at") or 0.0)
        new = decide(self.store.labels_by_code(since))
        old, self.tuning = self.tuning, new
        self.thresholds = tuned_thresholds(self.cfg.thresholds, new)
        if new != old:
            logger.info("sms_tuning_changed", steps=new)
            self._tuning_published_at = 0.0  # tell the learned models on the next tick
        if not announce:
            return
        now = time.time() if now is None else now
        for code in sorted(set(old) | set(new)):
            if old.get(code, 0) == new.get(code, 0):
                continue
            for c in self.cfg.technicians:
                lang = self._lang_for(c)
                name = self.t.render(lang, "names", code)
                text = (self.t.render(lang, "tuned", site=self.cfg.site_name, name=name,
                                      steps=new[code], max_steps=MAX_STEPS) if new.get(code)
                        else self.t.render(lang, "tuned_back", site=self.cfg.site_name, name=name))
                self._send(c.phone, text, "technician", "tuning", None, now)

    def _share_tuning(self, now: float) -> None:
        """Pass the steps for learned warnings to the inference container.

        Repeated every ten minutes so a restarted container catches up without
        the broker having to keep anything.
        """
        if self._publish_tuning is None or now - self._tuning_published_at < 600:
            return
        self._tuning_published_at = now
        self._publish_tuning({"at": now, "steps": {k: v for k, v in self.tuning.items()
                                                   if k in LEARNED_CODES}})

    # ── time helpers ────────────────────────────────────────────
    def _local_hour(self, now: float) -> int:
        return int(((now / 3600) + self.cfg.utc_offset_hours) % 24)

    def _local_day(self, now: float) -> int:
        return int((now + self.cfg.utc_offset_hours * 3600) // DAY)

    def _local_hhmm(self, now: float) -> str:
        return time.strftime("%H:%M", time.gmtime(now + self.cfg.utc_offset_hours * 3600))

    def _quiet(self, now: float) -> bool:
        if not self.cfg.quiet_hours:
            return False
        start, end = self.cfg.quiet_hours
        h = self._local_hour(now)
        return start <= h or h < end if start > end else start <= h < end

    # ── inputs ──────────────────────────────────────────────────
    def on_telemetry(self, payload: dict[str, Any]) -> None:
        try:
            ts = float(payload.get("timestamp", time.time() * 1000)) / 1000
            self.state.update(str(payload.get("asset_id") or "unknown"),
                              str(payload["metric_name"]), float(payload["value"]), ts)
            if payload["metric_name"] == "battery.soc_pct":
                v = float(payload["value"])
                self._min_soc_today = v if self._min_soc_today is None else min(self._min_soc_today, v)
        except (KeyError, TypeError, ValueError):
            logger.warning("sms_bad_telemetry")

    def on_findings(self, payload: dict[str, Any]) -> None:
        """Warnings from the on-site learned models (inference container)."""
        try:
            self._learned = [
                Fault(str(f["code"]), "warning", str(f.get("asset_id", "site")),
                      {k: v for k, v in (f.get("values") or {}).items()})
                for f in payload.get("findings", [])
            ]
            self._learned_at = float(payload.get("at", time.time()))
        except (KeyError, TypeError, ValueError):
            logger.warning("sms_bad_findings")

    def _learned_faults(self, now: float) -> list[Fault]:
        # findings are refreshed every 15 minutes; ignore them if they go stale
        if self._learned_at is None or now - self._learned_at > 3600:
            return []
        return list(self._learned)

    def on_box_alert(self, payload: dict[str, Any], now: float | None = None) -> None:
        now = time.time() if now is None else now
        kind = str(payload.get("type", "box"))
        if now - self._box_alert_last.get(kind, 0) < 3600:
            return
        self._box_alert_last[kind] = now
        detail = str(payload.get("message") or payload.get("reason") or kind)[:60]
        fault = Fault("box_alert", "critical", "box", {"detail": detail})
        self._alert(fault, now)

    # ── outbound ────────────────────────────────────────────────
    def _send(self, phone: str, text: str, role: str, kind: str, ref: str | None, now: float) -> None:
        ok = self.backend.send(phone, text)
        self.store.record_delivery(ref, phone, role, kind, ok, now, text)

    def _lang_for(self, contact: Contact) -> str:
        return self.store.get_lang(contact.phone) or contact.lang or self.cfg.default_lang

    def _alert(self, fault: Fault, now: float) -> None:
        ref = self.store.new_ref()
        self._refs[fault.key] = ref
        self.store.record_alert(ref, fault.code, fault.severity, fault.asset_id, fault.values, now)
        for c in self.cfg.technicians:
            lang = self._lang_for(c)
            body = self.t.render(lang, "faults", fault.code, **fault.values)
            suffix = self.t.render(lang, "technician_suffix", ref=ref)
            self._send(c.phone, f"{self.cfg.site_name}: {body}{suffix}", "technician", "alert", ref, now)
        if fault.code in self.cfg.notify_community_on:
            for c in self.cfg.community:
                self._send(c.phone, self._status_text(self._lang_for(c), now), "community",
                           "notice", ref, now)
        logger.info("sms_alert_sent", code=fault.code, ref=ref, severity=fault.severity)

    def _cleared(self, fault: Fault, now: float) -> None:
        self.store.clear_alert(fault.code, fault.asset_id, now)
        ref = self._refs.pop(fault.key, None)
        if fault.severity != "critical":
            return
        recipients = [(c, "technician") for c in self.cfg.technicians]
        if fault.code in self.cfg.notify_community_on:
            recipients += [(c, "community") for c in self.cfg.community]
        for c, role in recipients:
            lang = self._lang_for(c)
            what = self.t.render(lang, "recovered", fault.code)
            text = self.t.render(lang, "cleared", site=self.cfg.site_name, what=what)
            self._send(c.phone, text, role, "cleared", ref, now)

    # ── status ──────────────────────────────────────────────────
    def _status_text(self, lang: str, now: float) -> str:
        site = self.cfg.site_name
        th = self.thresholds
        fresh = self.state.last_seen_s is not None and now - self.state.last_seen_s <= th.comms_lost_s
        if not fresh:
            return self.t.render(lang, "status_unknown", site=site)
        if self._outage_since is not None:
            causes = [f for f in evaluate(self.state, th, now, self.cfg.rated_power_w)
                      if f.code != "power_off" and f.severity == "critical"]
            cause = (self.t.render(lang, "names", causes[0].code) if causes
                     else ("being checked" if lang == "en" else "dem dey check am"))
            return self.t.render(lang, "status_off", site=site,
                                 since=self._local_hhmm(self._outage_since), cause=cause)
        details = ""
        socs = self.state.find("battery.soc_pct", now, th.fresh_s)
        if socs:
            details += self.t.render(lang, "status_battery", soc=round(socs[0][1]))
        pv = self.state.find("solar.pv_power_w", now, th.fresh_s)
        if not pv:  # inverters that only report per-tracker power
            pv = [x for i in range(1, 5) for x in self.state.find(f"solar.mppt{i}_power_w", now, th.fresh_s)]
        if pv:
            details += self.t.render(lang, "status_solar", solar_kw=round(sum(v for _, v in pv) / 1000, 1))
        return self.t.render(lang, "status_on", site=site, details=details)

    # ── periodic work ───────────────────────────────────────────
    def tick(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        day = self._local_day(now)
        if self._day is not None and day != self._day:
            self._outage_seconds_today = 0.0
            self._min_soc_today = None
        self._day = day

        self._share_tuning(now)
        faults = evaluate(self.state, self.thresholds, now, self.cfg.rated_power_w)
        outage = any(f.code == "power_off" for f in faults)
        faults += self._learned_faults(now)
        if self._last_tick is not None and self._outage_since is not None:
            self._outage_seconds_today += now - self._last_tick
        if outage and self._outage_since is None:
            self._outage_since = now
        elif not outage:
            self._outage_since = None
        self._last_tick = now

        to_alert, cleared = self.tracker.step(faults, now)
        quiet = self._quiet(now)
        sent_now: set[str] = set()
        for f in to_alert:
            if f.severity == "warning" and quiet and f.code not in self.cfg.urgent_warnings:
                if all(key != f.key for _, key in self._deferred):
                    self._deferred.append((f, f.key))
            else:
                self._alert(f, now)
                sent_now.add(f.key)
        if not quiet and self._deferred:
            active = self.tracker.active
            for _, key in self._deferred:
                if key in active and key not in sent_now:
                    self._alert(active[key], now)
            self._deferred.clear()
        for f in cleared:
            self._cleared(f, now)

        if (self.cfg.daily_summary_hour is not None
                and self._local_hour(now) == self.cfg.daily_summary_hour
                and self._summary_sent_day != day):
            self._summary_sent_day = day
            self._daily_summary(now)

    def _daily_summary(self, now: float) -> None:
        start = (self._local_day(now) * DAY) - self.cfg.utc_offset_hours * 3600
        elapsed = max(now - start, 1.0)
        uptime = max(0.0, 100.0 * (1 - self._outage_seconds_today / elapsed))
        alerts = len(self.store.alerts_between(start, now))
        min_soc = f"{round(self._min_soc_today)}%" if self._min_soc_today is not None else "n/a"
        for c in self.cfg.technicians:
            text = self.t.render(self._lang_for(c), "summary", site=self.cfg.site_name,
                                 uptime=round(uptime), alerts=alerts, min_soc=min_soc)
            self._send(c.phone, text, "technician", "summary", None, now)

    def poll_inbox(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        for phone, text in self.backend.receive():
            reply = self.handle_inbound(phone, text, now)
            if reply:
                role = "technician" if normalise_phone(phone) in self._tech_phones else "community"
                self._send(phone, reply, role, "reply", None, now)

    def handle_inbound(self, phone: str, text: str, now: float) -> str | None:
        phone = normalise_phone(phone)
        cmd = parse(text)
        is_tech = phone in self._tech_phones
        lang = self.store.get_lang(phone) or next(
            (c.lang for c in self.cfg.technicians + self.cfg.community if c.phone == phone),
            self.cfg.default_lang,
        )
        self.store.record_inbound(phone, cmd.kind, now)

        if cmd.kind == "status":
            if not (self.cfg.public_status or is_tech):
                return None
            if self.store.inbound_count(phone, "status", now - 3600) > self.cfg.status_per_hour:
                return None
            return self._status_text(lang, now)

        if cmd.kind == "label":
            if not is_tech:
                return None
            ref = cmd.ref if cmd.ref and self.store.alert_exists(cmd.ref) else \
                self.store.latest_alert_ref_for(phone, now - self.cfg.label_window_s)
            if ref is None:
                return self.t.render(lang, "label_none")
            assert cmd.label is not None
            self.store.record_label(ref, cmd.label, cmd.note, phone, now)
            labels = self.t.raw(lang, "labels")
            label = labels.get(str(cmd.label), str(cmd.label)) if isinstance(labels, dict) else str(cmd.label)
            self._retune(announce=True, now=now)
            return self.t.render(lang, "label_ok", ref=ref, label=label)

        if cmd.kind == "reset":
            if not is_tech:
                return None
            self.store.set_meta("tuning_reset_at", repr(now))
            self._retune(announce=False)
            return self.t.render(lang, "tuning_reset")

        if cmd.kind == "lang" and cmd.lang in self.t.languages:
            self.store.set_lang(phone, cmd.lang)
            return self.t.render(cmd.lang, "lang_ok")

        if self.store.inbound_count(phone, cmd.kind, now - DAY) > self.cfg.help_per_day:
            return None
        return self.t.render(lang, "help")
