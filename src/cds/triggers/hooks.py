from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from cds.config.models import HookRuleConfig, HookTriggerConfig


def _payload_template_context(payload: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = dict(payload)
    bbox = payload.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        context["bbox_x1"] = bbox[0]
        context["bbox_y1"] = bbox[1]
        context["bbox_x2"] = bbox[2]
        context["bbox_y2"] = bbox[3]
        try:
            context["bbox_width"] = int(bbox[2]) - int(bbox[0])
            context["bbox_height"] = int(bbox[3]) - int(bbox[1])
        except Exception:
            pass
    return context


def _payload_env(payload: dict[str, Any]) -> dict[str, str]:
    context = _payload_template_context(payload)
    exported: dict[str, str] = {
        "CDS_EVENT_JSON": json.dumps(payload, ensure_ascii=True),
    }
    for key, value in context.items():
        env_key = f"CDS_{str(key).upper()}"
        if value is None:
            exported[env_key] = ""
        elif isinstance(value, (str, int, float, bool)):
            exported[env_key] = str(value)
    return exported


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

    def _render_command(self, command: list[str], payload: dict[str, Any]) -> list[str]:
        context = _payload_template_context(payload)
        rendered: list[str] = []
        for token in command:
            rendered.append(str(token).format_map(context))
        return rendered

    def _run(self, command: list[str], payload: dict[str, Any], timeout: float, payload_mode: str) -> None:
        env = os.environ.copy()
        stdin_data = None
        if payload_mode.lower() == "env":
            env.update(_payload_env(payload))
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

            try:
                command = self._render_command(rule.command, payload)
            except KeyError as exc:
                self._logger.warning(
                    "hook command template missing payload key=%s command=%s",
                    exc.args[0],
                    rule.command,
                )
                continue
            except ValueError as exc:
                self._logger.warning(
                    "hook command template invalid command=%s error=%s",
                    rule.command,
                    exc,
                )
                continue
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
