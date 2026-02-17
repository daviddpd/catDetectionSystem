from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from cds.config.models import HookRuleConfig, HookTriggerConfig


class HookTrigger:
    def __init__(self, config: HookTriggerConfig) -> None:
        self._config = config
        self._logger = logging.getLogger("cds.trigger.hook")
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._last_emit: dict[tuple[int, str], float] = {}

    def _is_allowed(self, command: list[str]) -> bool:
        if not command:
            return False
        if not self._config.allowlist:
            return False
        cmd_path = command[0]
        return cmd_path in self._config.allowlist

    def _run(self, command: list[str], payload: dict[str, Any], timeout: float, payload_mode: str) -> None:
        env = os.environ.copy()
        stdin_data = None
        if payload_mode.lower() == "env":
            env["CDS_EVENT_JSON"] = json.dumps(payload, ensure_ascii=True)
        else:
            stdin_data = json.dumps(payload, ensure_ascii=True)

        try:
            result = subprocess.run(
                command,
                input=stdin_data,
                text=True,
                capture_output=True,
                env=env,
                timeout=timeout,
                check=False,
            )
            self._logger.info(
                "hook command=%s rc=%d stdout=%s stderr=%s",
                command,
                result.returncode,
                result.stdout.strip(),
                result.stderr.strip(),
            )
        except subprocess.TimeoutExpired:
            self._logger.warning("hook timeout command=%s timeout=%.2fs", command, timeout)
        except Exception as exc:
            self._logger.warning("hook failed command=%s error=%s", command, exc)

    def emit(self, label: str, payload: dict[str, Any]) -> None:
        if not self._config.enabled:
            return

        for idx, rule in enumerate(self._config.rules):
            if rule.classes and label not in rule.classes:
                continue

            command = list(rule.command)
            if not self._is_allowed(command):
                self._logger.warning("hook blocked by allowlist command=%s", command)
                continue

            now = time.monotonic()
            key = (idx, label)
            last = self._last_emit.get(key, 0.0)
            if now - last < rule.cooldown_seconds:
                continue

            self._last_emit[key] = now
            self._executor.submit(
                self._run,
                command,
                payload,
                rule.timeout_seconds,
                rule.payload_mode,
            )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
