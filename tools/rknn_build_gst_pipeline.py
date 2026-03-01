#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _normalize_codec(codec: str | None, uri: str) -> str:
    if codec:
        value = codec.strip().lower()
    else:
        suffix = Path(uri).suffix.lower()
        if suffix in {".mp4", ".mov", ".m4v", ".mkv"}:
            value = "h264"
        else:
            value = "h264"
    if value not in {"h264", "h265", "hevc"}:
        raise ValueError(f"Unsupported codec '{codec}'. Expected h264 or h265.")
    return "h265" if value == "hevc" else value


def _build_gst_chain(uri: str, *, codec: str, source_kind: str, sink_kind: str) -> str:
    parser = "h264parse" if codec == "h264" else "h265parse"
    if source_kind == "rtsp":
        depay = "rtph264depay" if codec == "h264" else "rtph265depay"
        source = f"rtspsrc location={uri} latency=0 protocols=tcp ! {depay} ! {parser}"
    else:
        demux = "qtdemux" if Path(uri).suffix.lower() in {".mp4", ".mov", ".m4v"} else "matroskademux"
        source = f"filesrc location={uri} ! {demux} ! {parser}"

    tail = "fakesink sync=false"
    if sink_kind == "appsink":
        tail = "appsink drop=true sync=false max-buffers=1"

    return (
        f"{source} ! "
        "mppvideodec ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        f"{tail}"
    )


def _build_commands(uri: str, *, codec: str, source_kind: str) -> tuple[str, str]:
    appsink_pipeline = _build_gst_chain(uri, codec=codec, source_kind=source_kind, sink_kind="appsink")
    smoke_pipeline = _build_gst_chain(uri, codec=codec, source_kind=source_kind, sink_kind="fakesink")
    gst_launch = f"gst-launch-1.0 {smoke_pipeline}"
    return appsink_pipeline, gst_launch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build recommended Rockchip MPP GStreamer pipelines for CDS and gst-launch smoke tests.",
    )
    parser.add_argument("--uri", required=True, help="Input file path or RTSP URI")
    parser.add_argument(
        "--codec",
        choices=["h264", "h265", "hevc"],
        help="Video codec. Defaults to h264 unless specified.",
    )
    parser.add_argument(
        "--source-kind",
        choices=["auto", "file", "rtsp"],
        default="auto",
        help="Source type. Defaults to auto-detect from URI.",
    )
    parser.add_argument(
        "--format",
        choices=["all", "cds", "gst-launch"],
        default="all",
        help="Output format to print.",
    )
    args = parser.parse_args()

    uri = args.uri.strip()
    source_kind = args.source_kind
    if source_kind == "auto":
        source_kind = "rtsp" if uri.lower().startswith("rtsp://") else "file"

    codec = _normalize_codec(args.codec, uri)
    cds_pipeline, gst_launch = _build_commands(uri, codec=codec, source_kind=source_kind)

    if args.format in {"all", "cds"}:
        print(f"cds_pipeline={cds_pipeline}")
    if args.format in {"all", "gst-launch"}:
        print(f"gst_launch={gst_launch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
