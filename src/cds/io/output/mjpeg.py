from __future__ import annotations

import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2

from cds.io.output.base import OutputSink


class _FrameStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._frame: bytes | None = None
        self._version = 0

    def put(self, frame_bytes: bytes) -> None:
        with self._cv:
            self._frame = frame_bytes
            self._version += 1
            self._cv.notify_all()

    def wait_for_next(self, last_version: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
        with self._cv:
            if self._version <= last_version:
                self._cv.wait(timeout=timeout)
            return self._frame, self._version


class _StreamingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, host: str, port: int, path: str) -> None:
        self.frame_store = _FrameStore()
        self.stream_path = path
        self.running = True
        super().__init__((host, port), _StreamingHandler)


class _StreamingHandler(BaseHTTPRequestHandler):
    server: _StreamingHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        if self.path != self.server.stream_path:
            self.send_error(HTTPStatus.NOT_FOUND, "Stream endpoint not found")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.end_headers()

        version = -1
        while self.server.running:
            frame_bytes, version = self.server.frame_store.wait_for_next(version, timeout=1.0)
            if frame_bytes is None:
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame_bytes)
                self.wfile.write(b"\r\n")
            except Exception:
                return

    def log_message(self, fmt: str, *args: Any) -> None:
        return


class MjpegSink(OutputSink):
    def __init__(self, host: str, port: int, path: str = "/stream.mjpg") -> None:
        self._host = host
        self._port = port
        self._path = path if path.startswith("/") else f"/{path}"
        self._server: _StreamingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def endpoint_url(self) -> str:
        return f"http://{self._host}:{self._port}{self._path}"

    def open(self) -> None:
        if self._server is not None:
            return
        self._server = _StreamingHTTPServer(self._host, self._port, self._path)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def write(self, frame: Any, metadata: dict | None = None) -> bool:
        if self._server is None:
            self.open()

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
        )
        if not ok:
            return True

        assert self._server is not None
        self._server.frame_store.put(encoded.tobytes())
        return True

    def close(self) -> None:
        if self._server is None:
            return
        self._server.running = False
        self._server.shutdown()
        self._server.server_close()
        self._server = None

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def name(self) -> str:
        return "mjpeg"
