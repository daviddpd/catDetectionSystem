from __future__ import annotations

import queue
import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestFrameQueue(Generic[T]):
    """Bounded queue where latest frame wins by dropping oldest on full."""

    def __init__(self, maxsize: int = 2) -> None:
        self._queue: queue.Queue[T] = queue.Queue(maxsize=max(1, maxsize))
        self._lock = threading.Lock()

    def put_latest(self, item: T) -> int:
        dropped = 0
        with self._lock:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    dropped += 1
                except queue.Empty:
                    pass
            self._queue.put_nowait(item)
        return dropped

    def get(self, timeout: float = 0.1) -> T | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()
