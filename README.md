# catDetectionSystem

`catDetectionSystem` provides:
- Stage 1 low-latency detection runtime
- Stage 2 reproducible training, evaluation, active learning, and multi-target packaging

## TL;DR Quick Start

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Check host readiness.

```bash
./cds doctor --target all
```

3. Pull recommended baseline models for this host.

```bash
./cds train prefetch-models
```

4. Prepare/validate dataset and run training.

```bash
./cds dataset prepare --config config/dataset.yaml
./cds dataset validate --dataset-root dataset
./cds train --config config/train.yaml
./cds evaluate --config config/eval.yaml
./cds export --config config/export.yaml --targets all
```

## CLI Overview

Primary command: `./cds`

Runtime commands:
- `cds detect`
- `cds infer` (alias of `detect`)
- `cds monitor`
- `cds doctor --target runtime`

Training/model commands:
- `cds train`
- `cds evaluate`
- `cds export`
- `cds dataset prepare`
- `cds dataset validate`
- `cds doctor --target training`

Legacy compatibility alias:
- `cds detect-c4`

## Darknet (`opencv-darknet`) to Export Formats

Example legacy runtime command (your current `.cfg` + `.weights` pair):

```bash
./cds detect \
  --backend opencv-darknet \
  --cfg-path ./yolo/cfg/yolov-tiny-custom-416v6-64.cfg \
  --weights-path ./yolo/weights/yolov-tiny-custom-416v6-64_final.weights \
  --labels-path ./yolo/cfg/custom-names-v4.txt \
  --uri /path/to/video_or_rtsp
```

Important:
- `cds export` currently takes source models in `.pt` or `.onnx` format.
- direct `.cfg`/`.weights` export to all targets is not built into `cds export`.

Documented bridge flow:
1. Convert Darknet to ONNX with an external converter (example: [darknet-onnx](https://github.com/daviddpd/darknet-onnx)).
2. Use `cds export` from that ONNX for `onnx,rknn`.
3. For full multi-target output (`onnx,coreml,tensorrt,rknn`), use a `.pt` checkpoint and run `cds export --targets all`.

Full copy/paste commands: `docs/runbooks/darknet-legacy-conversion.md`

## Highest-Performance macOS Detect Command

On Apple Silicon, the highest-performance path in this project is typically CoreML with a `.mlpackage` export.

Export CoreML:

```bash
./cds export \
  --model artifacts/models/<run-id>/checkpoints/best.pt \
  --targets coreml \
  --output-dir artifacts/models/<run-id>
```

Run detect with CoreML:

```bash
./cds detect \
  --backend coreml \
  --model-path artifacts/models/<run-id>/exports/<model>.mlpackage \
  --labels-path dataset/classes.txt \
  --uri /path/to/video_or_rtsp
```

## Stage 2 Required Class Set

Minimum canonical classes:
- cat
- dog
- raccoon
- opossum
- fox
- squirrel
- bird
- skunk

## Bootstrap Data Source Notes

Example bootstrap command:

```bash
./cds train bootstrap-openvocab \
  --classes "cat,dog,raccoon,opossum,fox,squirrel,bird,skunk" \
  --source /path/to/camera_data
```

`--source` is typically your custom/local camera media root.

Directory sources are scanned recursively for supported media:
- images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`
- videos: `.mp4`, `.mkv`, `.avi`, `.mov`, `.m4v`, `.webm`

Bootstrap storage behavior:
- image sources are not copied into `artifacts`; absolute source paths are stored in review manifests
- labels are written under `artifacts/.../bootstrap/review/labels/`
- review manifests/config are written under `artifacts/.../bootstrap/review/`:
  - `predictions.jsonl`
  - `review_manifest.jsonl`
  - `review_config.json`
- optional: use `--materialize-non-image-frames` to save extracted JPEGs for video/stream inputs

Useful open datasets for augmentation/bootstrap:
- [Open Images V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/)
- [LILA Camera Trap datasets](https://lila.science/datasets/)
- [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps/)
- [NACTI](https://lila.science/datasets/nacti/)

## XML Annotation Notes

Stage 2 dataset prep accepts Pascal VOC XML as input and converts it to YOLO labels.

References:
- [PASCAL VOC](https://host.robots.ox.ac.uk/pascal/VOC/)
- [Pascal VOC XML format overview](https://roboflow.com/formats/pascal-voc-xml)

In this repo, XML is produced primarily by legacy annotation paths (`rtsp-object-ident.py` with XML options) or external labeling tools.

## CLIP Dependency for YOLO-World

YOLO-World bootstrap requires CLIP; this is included in `requirements.txt` as a VCS dependency:
- `clip @ git+https://github.com/ultralytics/CLIP.git`

## Config and Docs

Sample configs are under `config/`:
- `config/dataset.yaml`
- `config/train.yaml`
- `config/eval.yaml`
- `config/export.yaml`
- `config/runtime.yaml`

Detailed docs:
- `docs/STAGE2_TRAINING_AND_MODEL_PIPELINE.md`
- `docs/runbooks/darknet-legacy-conversion.md`
- `docs/runbooks/coreml.md`
- `docs/runbooks/tensorrt.md`
- `docs/runbooks/rknn.md`
