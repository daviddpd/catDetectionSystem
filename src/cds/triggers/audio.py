from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cds.config.models import AudioTriggerConfig


class AudioTrigger:
    def __init__(self, config: AudioTriggerConfig, max_workers: int = 2) -> None:
        self._config = config
        self._logger = logging.getLogger("cds.trigger.audio")
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._last_emit: dict[str, float] = {}
        self._player_cmd = self._resolve_player_command()

    def _resolve_player_command(self) -> str | None:
        system = platform.system().lower()
        if system == "darwin" and shutil.which("afplay"):
            return "afplay"

        if system == "linux":
            if shutil.which("paplay"):
                return "paplay"
            if shutil.which("aplay"):
                return "aplay"

        return None

    def _play(self, audio_path: str) -> None:
        if self._player_cmd is None:
            return
        try:
            subprocess.run(
                [self._player_cmd, audio_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except Exception as exc:
            self._logger.warning("audio trigger failed: %s", exc)

    def emit(self, label: str) -> None:
        if not self._config.enabled:
            return

        audio_path = self._config.class_to_audio.get(label)
        if not audio_path:
            return

        path = Path(audio_path)
        if not path.exists():
            return

        now = time.monotonic()
        last = self._last_emit.get(label, 0.0)
        if now - last < self._config.cooldown_seconds:
            return

        self._last_emit[label] = now
        self._executor.submit(self._play, str(path))

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
