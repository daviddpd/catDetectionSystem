# catDetectionSystem - Stage 1 Prompt (Inference and Runtime Refactor)

Date: February 16, 2026

## Role
You are an AI coding agent working in this repository. Implement a production-ready, cross-platform inference runtime for feral cat and urban wildlife detection.

Stage dependency:
- Run Stage 0 naming/spelling cleanup first. If Stage 0 is not complete, keep compatibility aliases for any renamed entrypoints.

## Hard Constraints
- Do not break existing behavior unless replacing it with an equivalent or better implementation.
- Preserve local window output by default.
- Keep frame latency low: if inference is slower than incoming frames, drop old frames instead of buffering.
- Keep architecture modular so new accelerators and models can be added later.
- Do not add Windows support in this stage.

## Current System Snapshot (what exists now)
- Main scripts are `/Users/dpd/Documents/projects/github/catDectionSystem/rtsp-object-ident.py` and `/Users/dpd/Documents/projects/github/catDectionSystem/object-ident.py`.
- Current detection path is OpenCV DNN + Darknet config/weights loading via `cv2.dnn.readNetFromDarknet(...)`.
- Current runtime uses threads and queues (`framesIn`, `framesOut`, `framesToWrite`), but frame queues are blocking and can accumulate latency.
- Video ingest is via OpenCV `VideoCapture(..., cv2.CAP_FFMPEG)` and manual `OPENCV_FFMPEG_CAPTURE_OPTIONS` env strings.
- Model and path config are mostly hard-coded.
- Audio alert (meow sound) is implemented with pygame.
- Annotation export exists via XML writing and helper tools in `/Users/dpd/Documents/projects/github/catDectionSystem/tools/`.

## Technology Decision (implement this)
- Keep YOLO family as the primary detection approach.
- Use Ultralytics as the primary model/runtime wrapper.
- Keep OpenCV for display, drawing overlays, and compatibility fallback.
- Keep Darknet support as legacy fallback only, not the primary future path.

Rationale:
- Ultralytics supports device selection and export targets needed for this project, including CoreML and RKNN exports, and stream inference controls for buffering.
- OpenCV DNN is useful as fallback and for image/video operations but is not a full training framework.
- Darknet remains useful for legacy cfg/weights but is less portable across Apple + Rockchip targets.

## Platform Targets and Accelerator Auto-Selection
Implement a backend selection module with this support policy:

1. macOS (Apple Silicon):
- `coreml` backend first (Core ML compute units auto)
- fallback to PyTorch/Ultralytics `mps`
- fallback to CPU

2. Linux + NVIDIA:
- Supported distro targets for delivery/testing:
  - Ubuntu LTS
  - RHEL/CentOS/Rocky family
- TensorRT engine (if artifact exists and runtime available)
- ONNX Runtime CUDA or PyTorch CUDA
- fallback to OpenCV CUDA/OpenCL if available
- fallback to CPU

3. Linux + Rockchip:
- Support standard Linux plus vendor or custom Linux distributions used for Rockchip NPUs.
- RKNN runtime backend (Toolkit2 families and old toolkit family where needed)
- fallback to CPU

4. Windows:
- Not supported in this stage.

Expose selected backend and reason at startup logs.

## Required Refactor Architecture
Create a modular package layout (names can vary, intent cannot):

- `detector/backends/`
- `detector/models/`
- `pipeline/`
- `io/ingest/`
- `io/output/`
- `monitoring/`
- `config/`

Entrypoint naming requirements:
- Do not reuse script names `rtsp-object-ident.py` or `object-ident.py` for the new runtime.
- Use project-reflective open source naming:
  - package: `cds`
  - primary CLI: `cds`
  - main commands:
    - `cds detect`
    - `cds monitor`
    - `cds doctor`

Define explicit interfaces:

- `DetectorBackend`:
  - `load(model_spec)`
  - `infer(frame) -> detections`
  - `warmup()`
  - `name()`
  - `device_info()`

- `VideoIngest`:
  - `open(uri, options)`
  - `read_latest()` non-blocking
  - must support RTSP, file, and directory image/video playback

- `OutputSink`:
  - local display window sink
  - remote streaming sink
  - structured event sink

## Frame-Drop and Throughput Requirements
Implement low-latency ingestion rules:

- Separate ingest thread/process from inference thread.
- Use bounded queue size 1 or 2 for "latest frame wins".
- On queue full, discard oldest frame before inserting newest.
- Track and report dropped frame count and effective FPS.
- Do not allow unbounded buffering between ingest and inference.

## Ingest and Decode Requirements
Keep FFmpeg-based ingest support and improve it:

- Auto-detect hardware decode capability at runtime.
- Prefer hardware decode when available:
  - Apple: VideoToolbox
  - NVIDIA: NVDEC/CUDA paths
  - Rockchip: platform decoder path where available
- Add a capability probe command at startup and print selected decoder path.
- Keep software decode fallback.

GStreamer vs PyAV decision for this stage:
- Do this evaluation now and implement the outcome.
- Decision:
  - Augment current OpenCV flow with a dedicated ingest layer.
  - Use PyAV as default advanced ingest/decode backend in Python.
  - Add optional GStreamer backend for Linux deployments where plugin pipelines are needed.
  - Keep OpenCV ingest as compatibility fallback.
- Required rationale in docs:
  - PyAV gives direct FFmpeg packet/frame control in Python.
  - GStreamer gives strong pipeline control and appsink queue/leaky options for live low-latency streams.
  - OpenCV remains simplest fallback and display integration path.

## Output Requirements
Keep existing local window output and add remote streaming option:

- Runtime modes:
  - default: interactive window on
  - headless: no interactive/video/audio output sinks

- Local:
  - existing OpenCV window behavior retained
  - overlays include class, confidence, backend, and FPS

- Remote:
  - implement at least one remote option:
    - RTSP re-stream, or
    - WebRTC, or
    - low-latency MJPEG endpoint
  - expose config toggle and endpoint URL in logs
  - must be disabled when running headless

- Headless flag:
  - add `--headless` (or equivalent)
  - must disable local window, audio playback, and remote video sinks
  - must suppress per-frame console chatter and standard detection event output
  - allow only critical startup/failure logs

## Detection Triggers
Implement a unified trigger subsystem with non-blocking actions:

1. Audio trigger:
- configurable mapping: class label -> audio file
- debounce/cooldown per class
- non-blocking playback
- replace pygame with lighter options:
  - preferred: command-based audio backend (`afplay` on macOS, `aplay`/`paplay` on Linux) via subprocess
  - optional Python backend plugin for systems without command tools

2. External action trigger:
- shell hook support for class/event-based actions
- execute action asynchronously in worker thread/process pool
- do not block ingest, inference, display, or write pipelines
- provide event payload to action as JSON via stdin or env var
- enforce timeout and capture stderr/stdout for auditing
- include opt-in safety controls:
  - allowlist command path(s)
  - rate limits/cooldowns
  - retry policy disabled by default unless configured

## Configuration
Use a configuration library that supports YAML, TOML, and JSON from one API.

Preferred choice:
- Dynaconf (supports TOML, YAML, JSON, env var layering, and overrides).

Add a single human-editable configuration system covering:

- model selection and path
- backend selection policy
- ingest URI and ingest options
- class filters
- detection trigger mappings (audio and shell hooks)
- output sinks
- monitoring settings

CLI flags should override config values.

## Monitoring and Observability
Implement:

- structured logs (JSON line option)
- periodic stats line (fps_in, fps_infer, dropped_frames, queue_depth, backend, decoder)
- optional Prometheus metrics endpoint (preferred)
- per-detection event stream (stdout JSON and optional file)

## Acceptance Criteria
- Runs on macOS Apple Silicon with auto-selected `coreml` or `mps` fallback.
- Runs on Linux NVIDIA (Ubuntu and RHEL/CentOS/Rocky targets) with CUDA path selected when available.
- Runs on Linux Rockchip targets with RKNN backend integration points and fallback behavior.
- Frame drop policy verified under overload: latency stays bounded.
- Local window works in default mode.
- `--headless` mode disables output sinks as specified.
- Remote stream works when enabled.
- Detection triggers work:
  - audio mapping
  - non-blocking external hook execution
- Existing class labels continue to work with new runtime.

## Suggested Implementation Sequence
1. Add config system and backend abstraction.
2. Add Ultralytics backend and keep OpenCV/Darknet fallback backend.
3. Replace queue logic with latest-frame policy and metrics.
4. Implement decoder capability probe and selected path logging.
5. Implement PyAV default ingest backend and optional GStreamer backend.
6. Add runtime modes and `--headless`.
7. Add remote output sink.
8. Add detection trigger subsystem (audio + shell hooks).
9. Add tests and smoke scripts.

## Validation Tasks
- Add a stress mode to simulate slow inference and verify dropped-frame behavior.
- Add integration smoke test against:
  - one RTSP URI
  - one local video file
  - one image directory
- Log selected backend and decoder in every run.

## External References (use these while implementing)
- Ultralytics train/predict/export docs: https://docs.ultralytics.com/modes/train/ and https://docs.ultralytics.com/modes/predict/ and https://docs.ultralytics.com/modes/export/
- Ultralytics YOLO-World docs: https://docs.ultralytics.com/models/yolo-world/
- OpenCV DNN API/backends: https://docs.opencv.org/4.x/d6/d0f/group__dnn.html
- OpenCV VideoCapture and backend notes: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
- Darknet build/runtime notes: https://github.com/AlexeyAB/darknet
- PyTorch MPS backend notes: https://docs.pytorch.org/docs/stable/notes/mps
- Core ML conversion guide: https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html
- FFmpeg protocol docs (RTSP/RTP): https://ffmpeg.org/ffmpeg-protocols.html
- PyAV docs: https://pyav.org/docs/stable/
- GStreamer queue and appsink docs: https://gstreamer.freedesktop.org/documentation/coreelements/queue.html and https://gstreamer.freedesktop.org/documentation/app/appsink.html
- Dynaconf docs: https://www.dynaconf.com/
- RKNN Toolkit2: https://github.com/airockchip/rknn-toolkit2
