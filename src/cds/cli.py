from __future__ import annotations

import argparse
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cds",
        description="catDetectionSystem modular runtime",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Run detection runtime")
    detect.add_argument("--config", help="Path to TOML/YAML/JSON config file")
    detect.add_argument("--uri", help="RTSP URI, local video path, image path, or directory")

    detect.add_argument(
        "--backend",
        choices=[
            "auto",
            "ultralytics",
            "coreml",
            "mps",
            "cpu",
            "cuda",
            "tensorrt",
            "opencv-darknet",
            "opencv",
            "darknet",
            "rknn",
        ],
        help="Override backend selector",
    )
    detect.add_argument(
        "--ingest-backend",
        choices=["auto", "pyav", "gstreamer", "opencv"],
        help="Override ingest backend selector",
    )

    detect.add_argument("--model-name", help="Model name label for logs")
    detect.add_argument("--model-path", help="Ultralytics model path (.pt/.onnx/.engine/.mlpackage/.rknn)")
    detect.add_argument("--cfg-path", help="Darknet cfg path for OpenCV fallback")
    detect.add_argument("--weights-path", help="Darknet weights path for OpenCV fallback")
    detect.add_argument("--labels-path", help="Label file path")
    detect.add_argument("--confidence", type=float, help="Confidence threshold")
    detect.add_argument("--nms", type=float, help="NMS threshold")
    detect.add_argument("--imgsz", type=int, help="Inference image size")
    detect.add_argument(
        "--class-filter",
        action="append",
        default=None,
        help="Restrict detections to class label (repeatable)",
    )

    detect.add_argument("--headless", action="store_true", help="Disable local/remote video and audio sinks")
    detect.add_argument("--window-name", help="OpenCV window title")
    detect.add_argument("--remote-mjpeg", action="store_true", help="Enable MJPEG output sink")
    detect.add_argument("--remote-host", help="MJPEG listen host")
    detect.add_argument("--remote-port", type=int, help="MJPEG listen port")
    detect.add_argument("--remote-path", help="MJPEG URL path")

    detect.add_argument("--queue-size", type=int, choices=[1, 2], help="Frame queue size (latest-frame policy)")
    detect.add_argument("--rate-limit-fps", type=float, help="Ingest frame rate cap")
    detect.add_argument("--gstreamer-pipeline", help="Optional full GStreamer pipeline string")
    detect.add_argument(
        "--pyav-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="PyAV/FFmpeg option pair, repeatable",
    )

    detect.add_argument("--json-logs", action="store_true", help="Emit structured JSON logs")
    detect.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Runtime log level")
    detect.add_argument("--quiet", action="store_true", help="Suppress non-warning logs")
    detect.add_argument("--prometheus", action="store_true", help="Enable Prometheus metrics endpoint")
    detect.add_argument("--prometheus-host", help="Prometheus bind host")
    detect.add_argument("--prometheus-port", type=int, help="Prometheus bind port")
    detect.add_argument("--event-file", help="Write per-detection JSON events to file")
    detect.add_argument("--no-event-stdout", action="store_true", help="Disable per-detection JSON events on stdout")

    detect.add_argument("--stress-sleep-ms", type=int, default=None, help="Artificial inference delay for frame-drop stress testing")

    monitor = subparsers.add_parser("monitor", help="Runtime environment monitor")
    monitor.add_argument("--interval", type=float, default=5.0, help="Polling interval seconds")
    monitor.add_argument("--count", type=int, default=0, help="Number of samples (0 = run forever)")
    monitor.add_argument("--json", action="store_true", help="Output JSON lines")

    doctor = subparsers.add_parser("doctor", help="Probe runtime and backend capabilities")
    doctor.add_argument("--config", help="Optional config file to evaluate")
    doctor.add_argument("--json", action="store_true", help="Emit JSON report")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]

    if args.command == "detect":
        from cds.commands.detect import run_detect

        return run_detect(args, repo_root)
    if args.command == "monitor":
        from cds.commands.monitor import run_monitor

        return run_monitor(args, repo_root)
    if args.command == "doctor":
        from cds.commands.doctor import run_doctor

        return run_doctor(args, repo_root)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
