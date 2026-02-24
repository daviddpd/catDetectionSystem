from __future__ import annotations

import argparse
from pathlib import Path


def _add_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to TOML/YAML/JSON config file")
    parser.add_argument("--uri", help="RTSP URI, local video path, image path, or directory")

    parser.add_argument(
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
    parser.add_argument(
        "--ingest-backend",
        choices=["auto", "pyav", "gstreamer", "opencv"],
        help="Override ingest backend selector",
    )

    parser.add_argument("--model-name", help="Model name label for logs")
    parser.add_argument("--model-path", help="Ultralytics model path (.pt/.onnx/.engine/.mlpackage/.rknn)")
    parser.add_argument("--cfg-path", help="Darknet cfg path for OpenCV fallback")
    parser.add_argument("--weights-path", help="Darknet weights path for OpenCV fallback")
    parser.add_argument("--labels-path", help="Label file path")
    parser.add_argument("--confidence", type=float, help="Confidence threshold")
    parser.add_argument(
        "--confidence-min",
        type=float,
        help="Higher confidence threshold used by benchmark frame export boundary logic",
    )
    parser.add_argument("--nms", type=float, help="NMS threshold")
    parser.add_argument("--imgsz", type=int, help="Inference image size")
    parser.add_argument(
        "--class-filter",
        action="append",
        default=None,
        help="Restrict detections to class label (repeatable)",
    )

    parser.add_argument("--headless", action="store_true", help="Disable local/remote video and audio sinks")
    parser.add_argument("--window-name", help="OpenCV window title")
    parser.add_argument("--remote-mjpeg", action="store_true", help="Enable MJPEG output sink")
    parser.add_argument("--remote-host", help="MJPEG listen host")
    parser.add_argument("--remote-port", type=int, help="MJPEG listen port")
    parser.add_argument("--remote-path", help="MJPEG URL path")
    parser.add_argument(
        "--export-frames",
        action="store_true",
        help="Benchmark mode only: export low-confidence candidate frames + VOC XML",
    )
    parser.add_argument(
        "--export-frames-dir",
        help="Directory for benchmark frame exports (JPEG + VOC XML)",
    )
    parser.add_argument(
        "--export-frames-sample-pct",
        type=float,
        help="Random sample percentage for in-band benchmark frame export (default 10)",
    )

    parser.add_argument("--queue-size", type=int, choices=[1, 2], help="Frame queue size (latest-frame policy)")
    parser.add_argument(
        "--rate-limit-fps",
        type=float,
        help="Sampling target FPS before inference (does not sleep RTSP ingest reads)",
    )
    parser.add_argument(
        "--clock",
        choices=["auto", "source", "asfast"],
        help="Input pacing policy: auto (live realtime, files encoded-rate), source, or asfast",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="File-only full-throttle mode: process every frame as fast as possible without dropping",
    )
    parser.add_argument("--gstreamer-pipeline", help="Optional full GStreamer pipeline string")
    parser.add_argument(
        "--pyav-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="PyAV/FFmpeg option pair, repeatable",
    )

    parser.add_argument("--json-logs", action="store_true", help="Emit structured JSON logs")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Runtime log level")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-warning logs")
    parser.add_argument("--prometheus", action="store_true", help="Enable Prometheus metrics endpoint")
    parser.add_argument("--prometheus-host", help="Prometheus bind host")
    parser.add_argument("--prometheus-port", type=int, help="Prometheus bind port")
    parser.add_argument("--event-file", help="Write per-detection JSON events to file")
    parser.add_argument("--no-event-stdout", action="store_true", help="Disable per-detection JSON events on stdout")

    parser.add_argument("--stress-sleep-ms", type=int, default=None, help="Artificial inference delay for frame-drop stress testing")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cds",
        description="catDetectionSystem runtime and training toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect = subparsers.add_parser("detect", help="Run detection runtime")
    _add_infer_args(detect)

    infer = subparsers.add_parser("infer", help="Alias of detect for config-driven inference")
    _add_infer_args(infer)

    train = subparsers.add_parser("train", help="Train models and run active learning workflows")
    train.add_argument("--config", help="Path to train config")
    train.add_argument("--model", help="Base checkpoint/model path")
    train.add_argument("--epochs", type=int, help="Training epochs")
    train.add_argument("--imgsz", type=int, help="Training image size")
    train.add_argument("--batch", type=int, help="Training batch size")
    train.add_argument("--device", choices=["cpu", "mps", "cuda"], help="Training device")
    train.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    train.add_argument("--dataset", help="Dataset YAML path")
    train.add_argument("--output-dir", help="Experiment output root directory")
    train.add_argument("--experiment-name", help="Experiment name")
    train.add_argument("--no-export", action="store_true", help="Skip export after training")
    train.add_argument("--export-targets", help="Comma-separated export targets")

    train_sub = train.add_subparsers(dest="train_command")

    prefetch = train_sub.add_parser(
        "prefetch-models",
        help="Detect best host engine and prefetch recommended baseline models",
    )
    prefetch.add_argument(
        "--output-dir",
        default="artifacts/models/prefetch",
        help="Optional directory for prefetch report",
    )

    bootstrap = train_sub.add_parser(
        "bootstrap-openvocab",
        help="Bootstrap pseudo-labels with YOLO-World",
    )
    bootstrap.add_argument("--classes", required=True, help="Comma-separated class prompts")
    bootstrap.add_argument(
        "--source",
        required=True,
        help="Source URI/path; directories are scanned recursively for supported media",
    )
    bootstrap.add_argument("--model", default="yolov8s-worldv2.pt", help="YOLO-World model")
    bootstrap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    bootstrap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    bootstrap.add_argument("--max-frames", type=int, default=0, help="Max frames/images (0 = all)")
    bootstrap.add_argument(
        "--materialize-non-image-frames",
        action="store_true",
        help="For video/stream sources, write extracted frame JPEGs into artifacts for review",
    )
    bootstrap.add_argument("--output-dir", default="artifacts/models", help="Artifact output directory")

    al = train_sub.add_parser("active-learning", help="Active learning queue and merge utilities")
    al_sub = al.add_subparsers(dest="active_command", required=True)

    al_queue = al_sub.add_parser("queue", help="Queue uncertain detections from runtime events")
    al_queue.add_argument("--events", required=True, help="Runtime event JSONL path")
    al_queue.add_argument("--output", required=True, help="Output queue JSONL path")
    al_queue.add_argument("--min-conf", type=float, default=0.30, help="Lower confidence bound")
    al_queue.add_argument("--max-conf", type=float, default=0.70, help="Upper confidence bound")
    al_queue.add_argument("--class-filter", help="Optional comma-separated class filter")
    al_queue.add_argument("--truth", help="Optional reviewed truth JSONL (frame_id + labels) for FP/FN queueing")

    al_merge = al_sub.add_parser("merge", help="Merge reviewed queue items into dataset split")
    al_merge.add_argument("--queue", required=True, help="Queue JSONL path")
    al_merge.add_argument("--source-images", required=True, help="Base directory for source images")
    al_merge.add_argument("--dataset", required=True, help="Target dataset root")
    al_merge.add_argument("--split", default="train", choices=["train", "val", "test"], help="Target split")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate trained model and apply promotion gating")
    evaluate.add_argument("--config", help="Evaluation config path")
    evaluate.add_argument("--model", help="Checkpoint path")
    evaluate.add_argument("--dataset", help="Dataset YAML path")
    evaluate.add_argument("--split", choices=["train", "val", "test"], help="Dataset split")
    evaluate.add_argument("--device", choices=["cpu", "mps", "cuda"], help="Evaluation device")
    evaluate.add_argument("--imgsz", type=int, help="Evaluation image size")
    evaluate.add_argument("--batch", type=int, help="Evaluation batch size")
    evaluate.add_argument("--conf", type=float, help="Evaluation confidence threshold")

    export = subparsers.add_parser("export", help="Export model artifacts for supported targets")
    export.add_argument("--config", help="Export config path")
    export.add_argument("--model", help="Source model checkpoint path")
    export.add_argument("--output-dir", help="Export artifact directory")
    export.add_argument("--targets", default="all", help="Comma-separated targets (or 'all')")
    export.add_argument("--imgsz", type=int, help="Export image size")
    export.add_argument("--half", action="store_true", help="Enable half precision where supported")

    dataset = subparsers.add_parser("dataset", help="Dataset conversion, split, and validation")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)

    ds_prepare = dataset_sub.add_parser("prepare", help="Convert XML to YOLO labels, split, and validate")
    ds_prepare.add_argument("--config", help="Dataset config path")
    ds_prepare.add_argument("--xml-root", help="Input XML annotation root")
    ds_prepare.add_argument("--image-root", help="Input image root")
    ds_prepare.add_argument("--output-root", help="Output dataset root")
    ds_prepare.add_argument("--classes", help="Comma-separated canonical class list")
    ds_prepare.add_argument(
        "--split-mode",
        choices=["deterministic", "time-aware"],
        help="Split strategy",
    )

    ds_validate = dataset_sub.add_parser("validate", help="Validate YOLO dataset and emit health report")
    ds_validate.add_argument("--config", help="Dataset config path")
    ds_validate.add_argument("--dataset-root", help="Dataset root path")
    ds_validate.add_argument("--classes", help="Comma-separated canonical class list")
    ds_validate.add_argument("--report", help="Output report JSON path")

    monitor = subparsers.add_parser("monitor", help="Runtime environment monitor")
    monitor.add_argument("--interval", type=float, default=5.0, help="Polling interval seconds")
    monitor.add_argument("--count", type=int, default=0, help="Number of samples (0 = run forever)")
    monitor.add_argument("--json", action="store_true", help="Output JSON lines")

    doctor = subparsers.add_parser("doctor", help="Probe runtime or training toolchain capabilities")
    doctor.add_argument("--config", help="Optional config file to evaluate")
    doctor.add_argument("--target", choices=["runtime", "training", "all"], default="runtime", help="Doctor scope")
    doctor.add_argument("--json", action="store_true", help="Emit JSON report")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]

    if args.command == "detect":
        from cds.commands.detect import run_detect

        return run_detect(args, repo_root)
    if args.command == "infer":
        from cds.commands.infer import run_infer

        return run_infer(args, repo_root)
    if args.command == "train":
        from cds.commands.train import run_train

        return run_train(args, repo_root)
    if args.command == "evaluate":
        from cds.commands.evaluate import run_evaluate

        return run_evaluate(args, repo_root)
    if args.command == "export":
        from cds.commands.export import run_export

        return run_export(args, repo_root)
    if args.command == "dataset":
        from cds.commands.dataset import run_dataset

        return run_dataset(args, repo_root)
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
