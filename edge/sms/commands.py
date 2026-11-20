"""Parse inbound SMS into commands."""

from __future__ import annotations

import re
from dataclasses import dataclass

STATUS_WORDS = {"POWER", "LIGHT", "STATUS", "WUTA", "NEPA", "LIGHT?", "POWER?"}
HELP_WORDS = {"HELP", "?", "INFO"}
_LABEL = re.compile(r"^\s*([123])\b\s*(?:#?\s*([A-Z0-9]{4})\b)?\s*(.*)$", re.IGNORECASE)
_RESET = re.compile(r"^\s*RESET(?:\s+(?:TUNING|ALERTS?))?\s*$", re.IGNORECASE)
_LANG = re.compile(r"^\s*LANG(?:UAGE)?\s+([A-Z]{2,3})\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Command:
    kind: str  # status | label | lang | reset | help | unknown
    label: int | None = None
    ref: str | None = None
    note: str = ""
    lang: str | None = None


def parse(text: str) -> Command:
    cleaned = " ".join(text.strip().split())
    upper = cleaned.upper()
    if not upper:
        return Command("unknown")
    if upper.split(" ")[0] in STATUS_WORDS:
        return Command("status")
    if upper in HELP_WORDS:
        return Command("help")
    if _RESET.match(cleaned):
        return Command("reset")
    m = _LANG.match(cleaned)
    if m:
        return Command("lang", lang=m.group(1).lower())
    m = _LABEL.match(cleaned)
    if m:
        ref = m.group(2).upper() if m.group(2) else None
        return Command("label", label=int(m.group(1)), ref=ref, note=m.group(3).strip())
    return Command("unknown")
