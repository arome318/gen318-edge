"""Entry point for the gen318-sms container.

    SMS_CONFIG_PATH=/config/sms.yaml SMS_DB_PATH=/data/sms/sms.db python -m sms.main
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import structlog

from common.mqtt_client import (
    TOPIC_ALERTS_CRITICAL,
    TOPIC_SITELEARN,
    TOPIC_SITELEARN_TUNING,
    TOPIC_TELEMETRY,
    MQTTClient,
)
from sms.backends import make_backend
from sms.service import SmsConfig, SmsService
from sms.store import SmsStore

logger = structlog.get_logger()


async def main() -> None:
    cfg = SmsConfig.from_yaml(Path(os.environ.get("SMS_CONFIG_PATH", "/config/sms.yaml")).read_text())
    db_path = os.environ.get("SMS_DB_PATH", "/data/sms/sms.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    mqtt = MQTTClient(
        host=os.environ.get("MQTT_HOST", "mosquitto"),
        port=int(os.environ.get("MQTT_PORT", "1883")),
        client_id="gen318-sms",
    )
    service = SmsService(cfg, make_backend(cfg.backend), SmsStore(db_path),
                         publish_tuning=lambda payload: mqtt.publish(TOPIC_SITELEARN_TUNING, payload))
    mqtt.connect()
    mqtt.subscribe(TOPIC_TELEMETRY, service.on_telemetry)
    mqtt.subscribe(TOPIC_ALERTS_CRITICAL, service.on_box_alert)
    mqtt.subscribe(TOPIC_SITELEARN, service.on_findings)
    tick_s = float(os.environ.get("SMS_TICK_SECONDS", "15"))
    logger.info("sms_started", site=cfg.site_name, technicians=len(cfg.technicians),
                community=len(cfg.community), backend=cfg.backend.get("type"))
    while True:
        try:
            await asyncio.to_thread(service.tick)
            await asyncio.to_thread(service.poll_inbox)
        except Exception:
            logger.exception("sms_loop_error")
        await asyncio.sleep(tick_s)


if __name__ == "__main__":
    asyncio.run(main())
