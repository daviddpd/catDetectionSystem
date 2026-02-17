# catDetectionSystem

`catDetectionSystem` is a modular feral cat and urban wildlife detection runtime.

Stage 1 (February 16, 2026) introduces a new production-oriented runtime package and CLI:
- package: `cds` (`src/cds`)
- commands:
  - `cds detect`
  - `cds monitor`
  - `cds doctor`

Legacy scripts remain in place for compatibility:
- `rtsp-object-ident.py`
- `object-ident.py`
- `run.sh`
- `run-c4.sh`

## Quick Start

1. Install dependencies.
2. Run a capability probe:

```bash
./cds doctor
```

3. Run detection:

```bash
./cds detect --uri /path/to/video.mp4 --headless
```

4. Interactive local window mode (default):

```bash
./cds detect --uri rtsp://user:pass@camera/stream --model-path /path/to/model.pt
```

## Runtime Architecture

The Stage 1 runtime is organized to keep ingest, inference, and outputs decoupled:
- `src/cds/detector/backends/`: detector backends and selection policy
- `src/cds/detector/models/`: model specification
- `src/cds/pipeline/`: runtime loop and low-latency queueing
- `src/cds/io/ingest/`: PyAV/GStreamer/OpenCV ingest backends
- `src/cds/io/output/`: local display, MJPEG sink, JSON event sink
- `src/cds/monitoring/`: structured logging, stats, Prometheus metrics
- `src/cds/config/`: Dynaconf-first config loading and defaults

## Backend Selection Policy

`cds detect` auto-selects inference backend using platform policy and logs both the selected backend and the reason:

1. macOS Apple Silicon:
- CoreML artifact path when provided (`.mlpackage`/`.mlmodel`)
- fallback to Ultralytics on `mps`
- fallback to CPU

2. Linux + NVIDIA:
- TensorRT engine path when available (`.engine` + TensorRT runtime)
- fallback to CUDA
- fallback to OpenCV GPU path (CUDA/OpenCL)
- fallback to CPU

3. Linux + Rockchip:
- RKNN integration path
- fallback to CPU

4. Windows:
- not supported in this stage

## Ingest and Decode

The ingest layer supports:
- RTSP streams
- local video files
- image files/directories

Low-latency behavior:
- bounded ingest->infer queue (`size=1` or `size=2`)
- latest-frame-wins policy
- oldest frame dropped on queue full
- dropped-frame count and effective FPS tracked in stats

Decoder probing:
- startup probes hardware decode capabilities
- selected decoder path and rationale are logged every run

### PyAV vs GStreamer vs OpenCV (Stage 1 decision)

This stage adopts:
- PyAV as the default advanced ingest/decode backend
- optional GStreamer backend for Linux pipeline control cases
- OpenCV ingest as compatibility fallback

Rationale:
- PyAV provides direct FFmpeg packet/frame control in Python.
- GStreamer supports explicit pipeline and queue/leaky control (`appsink`) on Linux deployments.
- OpenCV remains the simplest fallback and integrates directly with display overlays.

## Output Modes

Default mode:
- local OpenCV window enabled
- overlays include class, confidence, backend, and FPS

Headless mode (`--headless`):
- local window disabled
- audio trigger disabled
- remote video sink disabled
- stdout event stream disabled

Remote output:
- optional MJPEG sink (`--remote-mjpeg`) with endpoint logged at startup

## Triggers

Stage 1 trigger subsystem is non-blocking:
- audio trigger: class -> audio file map with per-class cooldown
- external action hooks: class/event-based shell commands

External hook safety controls:
- command allowlist
- cooldown/rate limiting
- timeout
- stdout/stderr audit logging

## Configuration

Configuration supports TOML/YAML/JSON through a Dynaconf-first loader.

Search order (when `--config` is not provided):
- `cds.toml`
- `cds.yaml` / `cds.yml`
- `cds.json`
- `settings.toml` / `settings.yaml` / `settings.yml` / `settings.json`

CLI flags override config values.

A complete sample file is provided at:
- `cds.toml.example`

## Monitoring and Observability

- structured logs with `--json-logs`
- periodic stats line includes:
  - `fps_in`
  - `fps_infer`
  - `dropped_frames`
  - `queue_depth`
  - `backend`
  - `decoder`
- optional Prometheus endpoint (`--prometheus`)
- per-detection JSON events to stdout and/or file

## Validation Utilities

- stress mode (artificial inference delay):

```bash
./cds detect --uri /path/to/video.mp4 --stress-sleep-ms 250 --headless
```

- smoke script for RTSP/file/directory:

```bash
PYTHONPATH=src python3 tools/smoke_stage1.py \
  --rtsp-uri rtsp://user:pass@camera/stream \
  --video-file /path/to/video.mp4 \
  --image-dir /path/to/images \
  --model-path /path/to/model.pt
```

## Legacy Compatibility

The top-level `cds` wrapper now routes to the Stage 1 Python CLI for `detect`, `monitor`, and `doctor`.

Compatibility alias retained:
- `cds detect-c4` (deprecated)
