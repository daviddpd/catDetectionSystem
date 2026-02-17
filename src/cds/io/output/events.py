from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class JsonEventSink:
    def __init__(self, stdout_enabled: bool, file_path: str | None = None) -> None:
        self._stdout_enabled = stdout_enabled
        self._file_path = Path(file_path).expanduser().resolve() if file_path else None
        self._file_handle = None
        self._lock = threading.Lock()

    def open(self) -> None:
        if self._file_path is not None:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = self._file_path.open("a", encoding="utf-8")

    def emit(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=True)
        with self._lock:
            if self._stdout_enabled:
                print(payload, flush=True)
            if self._file_handle is not None:
                self._file_handle.write(payload + "\n")
                self._file_handle.flush()

    def close(self) -> None:
        with self._lock:
            if self._file_handle is not None:
                self._file_handle.close()
                self._file_handle = None
