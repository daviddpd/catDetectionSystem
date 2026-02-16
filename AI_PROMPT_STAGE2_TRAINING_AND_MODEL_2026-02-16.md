# catDetectionSystem - Stage 2 Prompt (Training and Model Pipeline)

Date: February 16, 2026

## Role
You are an AI coding agent continuing from Stage 1. Build a reproducible training and model-packaging pipeline for urban animal detection.

Stage dependency:
- Stage 0 naming/spelling cleanup complete.
- Stage 1 runtime refactor complete enough to load exported artifacts.

## Goal
Produce a trainable system that starts from an open-source base model and supports custom fine-tuning from local camera data.

## Required Target Classes (minimum)
- cat
- dog
- raccoon
- opossum
- fox
- squirrel
- bird
- skunk

## Model Strategy (implement this)
Use a two-track approach:

1. Bootstrap detector:
- Use open-vocabulary YOLO-World as a quick baseline for required class prompts.
- Class prompt list must include all required animals above.
- Build user-friendly wrappers so this mode is one command:
  - `cds train bootstrap-openvocab --classes "cat,dog,raccoon,opossum,fox,squirrel,bird,skunk" --source <uri-or-path>`
- Save bootstrap predictions in a reviewable format suitable for curation and relabeling.

2. Production detector:
- Fine-tune a standard Ultralytics YOLO model (small/medium variant configurable) on the project-specific 8+ class dataset.
- This becomes the default runtime model once validation passes.

## Dataset and Label Pipeline
Build a repeatable dataset workflow:

- Input sources:
  - existing camera clips/images
  - existing XML annotations produced by current scripts

- Conversion:
  - do not force reuse of `/Users/dpd/Documents/projects/github/catDetectionSystem/tools/xml2txt.py`
  - reuse it only if it is the cleanest path; otherwise replace with a new converter
  - generate YOLO label txt files and data YAML

- Validation:
  - verify class names map exactly to canonical class IDs
  - detect out-of-bounds boxes
  - detect empty labels and duplicates
  - produce dataset health report

- Splits:
  - deterministic train/val/test split
  - optional time-aware split by date/source camera to reduce leakage

## Training CLI / Flag Design
Create a clear CLI entrypoint, for example:

- `cds train --config config/train.yaml`
- `cds infer --config config/runtime.yaml`
- `cds export --config config/export.yaml`
- `cds doctor --target training`

Training flags must include:
- model checkpoint/base model
- epochs, image size, batch size
- device selection (`cpu`, `mps`, `cuda`)
- mixed precision toggle
- dataset path
- experiment output directory
- export targets after training

## Toolchain Preflight (required)
Implement a `doctor` command that verifies and reports toolchain readiness before conversion:

- Python/runtime:
  - `ultralytics`
  - `onnx`
  - `onnxruntime` variants (cpu/cuda where relevant)
  - `coremltools` (macOS path)
- Platform conversion toolchains:
  - TensorRT components on NVIDIA hosts
  - RKNN toolkit family components on Rockchip conversion hosts
- System tools:
  - ffmpeg availability

The command must provide:
- pass/fail per dependency
- install hints for missing components
- which export targets are currently possible on this host

## Hardware Scope for Training
Training support expectations:

- Primary:
  - Linux + NVIDIA CUDA
  - macOS Apple Silicon using MPS (for lighter training/fine-tune)

- Optional/limited:
  - Rockchip platforms may run inference only; training not required there.

## Export and Packaging Requirements
After successful training, export these artifacts (when supported):

- PyTorch checkpoint
- ONNX
- CoreML package
- TensorRT engine (where available)
- RKNN conversion input artifact and conversion script flow

Store all artifacts in a predictable versioned directory layout.

Also deliver:
- conversion runbooks with copy/paste commands
- a single "export all available targets" command that skips unsupported targets with clear warnings

## RKNN Path Requirements
Add tooling and docs for converting ONNX to RKNN for:
- RK3588, RK3576, RK3566, RK3568, RK3562, RV1103, RV1106, RV1103B, RV1106B, RV1126B, RK2118 (Toolkit2 family)
- RK1808, RV1109, RV1126, RK3399Pro (legacy family path documentation)

Do not block Stage 2 completion if conversion cannot execute on host machine.

Provide detailed conversion instructions:
- include per-chip-family notes for toolkit2 vs legacy toolkit flows
- include calibration/quantization input expectations
- include artifact naming conventions and output paths
- include troubleshooting section for common conversion failures

## Evaluation and Gating
Add evaluation reports with:

- per-class precision/recall and mAP
- confusion matrix for required classes
- day/night or low-light subgroup metrics (if data available)
- latency benchmarks from Stage 1 runtime

Promotion rule:
- model is "production candidate" only if all required classes have acceptable recall targets defined in config.

## Active Learning Loop
Implement a loop for incremental improvement:

- capture false positives/false negatives from runtime events
- queue uncertain detections for review
- support relabeling and retraining without manual directory surgery

## Deliverables
1. Training pipeline code and configs.
2. Dataset validation/report command.
3. Toolchain `doctor` command for conversion readiness checks.
4. Export pipeline for multi-backend artifacts.
5. Documentation for training, evaluating, and promoting a model.
6. Conversion runbooks:
   - CoreML runbook
   - TensorRT runbook
   - RKNN runbook
7. Example command set for:
   - bootstrap open-vocabulary run
   - full fine-tune run
   - export run
   - preflight/doctor run

## Example Commands (must work or be clearly marked as host-specific)
- `cds doctor --target training`
- `cds train bootstrap-openvocab --classes "cat,dog,raccoon,opossum,fox,squirrel,bird,skunk" --source /data/camera`
- `cds train --config config/train.yaml`
- `cds evaluate --config config/eval.yaml`
- `cds export --config config/export.yaml --targets onnx,coreml,tensorrt,rknn`

## Non-Goals
- Do not require training on every deployment target.
- Do not couple training code tightly to runtime inference code.

## External References (use these while implementing)
- Ultralytics train docs: https://docs.ultralytics.com/modes/train/
- Ultralytics export docs (including RKNN/CoreML): https://docs.ultralytics.com/modes/export/
- Ultralytics YOLO-World docs: https://docs.ultralytics.com/models/yolo-world/
- Core ML conversion reference: https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html
- PyTorch MPS notes: https://docs.pytorch.org/docs/stable/notes/mps
- RKNN Toolkit2 repo: https://github.com/airockchip/rknn-toolkit2
- RKNN legacy toolkit repo: https://github.com/airockchip/rknn-toolkit
- RK NPU repos: https://github.com/airockchip/rknpu and https://github.com/airockchip/RK3399Pro_npu
- ONNX Runtime docs: https://onnxruntime.ai/docs/
