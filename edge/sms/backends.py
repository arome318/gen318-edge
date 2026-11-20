"""SMS transports.

``ATModemBackend`` talks to any GSM/LTE modem that speaks standard text-mode
AT commands (SIM800, SIM7600, Quectel EC25 and most USB dongles) over a
serial port. No internet is used: messages go over the mobile network's SMS
channel, which works on 2G.

``DryRunBackend`` writes outgoing messages to a JSONL file and reads incoming
ones from another, for demos, tests and bench work without a SIM.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import structlog

logger = structlog.get_logger()


class SmsBackend(Protocol):
    def send(self, phone: str, text: str) -> bool: ...
    def receive(self) -> list[tuple[str, str]]: ...


class DryRunBackend:
    def __init__(self, directory: str) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.outbox = self.dir / "outbox.jsonl"
        self.inbox = self.dir / "inbox.jsonl"
        self._offset_file = self.dir / ".inbox_offset"
        self._lock = threading.Lock()

    def send(self, phone: str, text: str) -> bool:
        with self._lock, self.outbox.open("a") as fh:
            fh.write(json.dumps({"to": phone, "text": text, "at": time.time()}) + "\n")
        logger.info("sms_dry_run_sent", to=phone[-4:], chars=len(text))
        return True

    def receive(self) -> list[tuple[str, str]]:
        with self._lock:
            if not self.inbox.exists():
                return []
            offset = int(self._offset_file.read_text()) if self._offset_file.exists() else 0
            with self.inbox.open() as fh:
                fh.seek(offset)
                lines = fh.readlines()
                self._offset_file.write_text(str(fh.tell()))
        out = []
        for line in lines:
            try:
                msg = json.loads(line)
                out.append((str(msg["from"]), str(msg["text"])))
            except (ValueError, KeyError):
                logger.warning("sms_dry_run_bad_inbox_line")
        return out


_CMGL = re.compile(r'^\+CMGL:\s*(\d+),"[^"]*","([^"]*)"')


class ATModemBackend:
    """Text-mode AT command SMS over a serial port."""

    def __init__(self, port: str, baudrate: int = 115200, serial_factory: Any = None) -> None:
        self.port = port
        self.baudrate = baudrate
        self._factory = serial_factory
        self._ser: Any = None
        self._lock = threading.Lock()

    def _open(self) -> Any:
        if self._ser is None:
            if self._factory is None:
                import serial  # pyserial

                self._ser = serial.Serial(self.port, self.baudrate, timeout=1)
            else:
                self._ser = self._factory()
            for cmd in ("AT", "ATE0", "AT+CMGF=1", 'AT+CSCS="GSM"'):
                self._cmd(cmd)
        return self._ser

    def _read_until(self, terminators: tuple[str, ...], timeout: float) -> str:
        buf = ""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            chunk = self._ser.read(256)
            if chunk:
                buf += chunk.decode("ascii", "replace")
                if any(t in buf for t in terminators):
                    return buf
            else:
                time.sleep(0.05)
        raise TimeoutError(f"modem timeout waiting for {terminators}: {buf[-80:]!r}")

    def _cmd(self, cmd: str, timeout: float = 5.0) -> str:
        self._ser.write((cmd + "\r").encode("ascii"))
        resp = self._read_until(("\r\nOK\r\n", "ERROR"), timeout)
        if "ERROR" in resp:
            raise RuntimeError(f"modem error for {cmd}: {resp.strip()[-80:]}")
        return resp

    def send(self, phone: str, text: str) -> bool:
        with self._lock:
            try:
                self._open()
                self._ser.write(f'AT+CMGS="{phone}"\r'.encode("ascii"))
                self._read_until((">",), 10)
                payload = text.encode("ascii", "replace") + b"\x1a"
                self._ser.write(payload)
                self._read_until(("+CMGS", "ERROR"), 60)
                return True
            except Exception:
                logger.exception("sms_send_failed", to=phone[-4:])
                self._reset()
                return False

    def receive(self) -> list[tuple[str, str]]:
        with self._lock:
            try:
                self._open()
                resp = self._cmd('AT+CMGL="ALL"', timeout=15)
            except Exception:
                logger.exception("sms_receive_failed")
                self._reset()
                return []
            messages, indexes = parse_cmgl(resp)
            for idx in indexes:
                try:
                    self._cmd(f"AT+CMGD={idx}")
                except Exception:
                    logger.warning("sms_delete_failed", index=idx)
            return messages

    def _reset(self) -> None:
        try:
            if self._ser is not None:
                self._ser.close()
        finally:
            self._ser = None


def parse_cmgl(resp: str) -> tuple[list[tuple[str, str]], list[int]]:
    """Parse an AT+CMGL text-mode listing into (sender, text) pairs."""
    messages: list[tuple[str, str]] = []
    indexes: list[int] = []
    lines = [ln for ln in resp.replace("\r", "").split("\n")]
    i = 0
    while i < len(lines):
        m = _CMGL.match(lines[i].strip())
        if m:
            indexes.append(int(m.group(1)))
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("+CMGL:") and lines[i].strip() != "OK":
                body.append(lines[i])
                i += 1
            messages.append((m.group(2), "\n".join(body).strip()))
            continue
        i += 1
    return messages, indexes


def make_backend(cfg: dict[str, Any]) -> SmsBackend:
    kind = (cfg or {}).get("type", "dry_run")
    if kind == "at_modem":
        return ATModemBackend(cfg.get("port", "/dev/ttyUSB2"), int(cfg.get("baudrate", 115200)))
    if kind == "dry_run":
        return DryRunBackend(cfg.get("directory", "/data/sms"))
    raise ValueError(f"unknown SMS backend type: {kind}")
