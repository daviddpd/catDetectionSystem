# Stage 2 Training and Model Pipeline

This document describes the Stage 2 training/evaluation/export workflow for `catDetectionSystem`.

## TL;DR (Getting Started)

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Check runtime + training readiness.

```bash
./cds doctor --target all
```

If `doctor --target training` reports missing `clip`, install:

```bash
pip install "clip @ git+https://github.com/ultralytics/CLIP.git"
```

3. Prefetch recommended baseline models for this host (engine-aware recommendation).

```bash
./cds train prefetch-models
```

4. Build dataset from XML + images and validate.

```bash
./cds dataset prepare --config config/dataset.yaml
./cds dataset validate --dataset-root dataset
```

5. Run bootstrap open-vocabulary labeling for fast curation.

```bash
./cds train bootstrap-openvocab \
  --classes "cat,dog,raccoon,opossum,fox,squirrel,bird,skunk" \
  --source /path/to/camera_data
```

6. Fine-tune and export.

```bash
./cds train --config config/train.yaml
./cds evaluate --config config/eval.yaml
./cds export --config config/export.yaml --targets all
```

## Canonical Class Set

Minimum required classes:
- cat
- dog
- raccoon
- opossum
- fox
- squirrel
- bird
- skunk

## Data Sources: Custom vs Open Data

`--source /path/to/camera_data` is expected to be your local/custom media root by default.

The bootstrap command now recursively scans directory sources for supported media:
- images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`
- videos: `.mp4`, `.mkv`, `.avi`, `.mov`, `.m4v`, `.webm`

Open-source datasets that are useful for bootstrapping and augmentation:
- Ultralytics Open Images V7 integration: [Open Images V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/)
- Camera-trap catalog (LILA): [LILA Datasets](https://lila.science/datasets/)
- Caltech Camera Traps (contains many urban wildlife scenarios): [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps/)
- NACTI (North American Camera Trap Images): [NACTI](https://lila.science/datasets/nacti/)

Notes:
- Use custom camera data as the primary source for deployment realism.
- Open datasets are useful for class coverage and pre-curation, but label taxonomies vary.

## Bootstrap Open-Vocabulary Flow

Use YOLO-World to quickly generate reviewable pseudo-labels:

```bash
./cds train bootstrap-openvocab \
  --classes "cat,dog,raccoon,opossum,fox,squirrel,bird,skunk" \
  --source /path/to/camera_data
```

The class list must include all required classes above.

### Bootstrap Output Files and Formats

Outputs are written to a versioned run folder:
- `artifacts/models/<run-id>/bootstrap/review/images/`
  - extracted review frames/images (`.jpg`)
- `artifacts/models/<run-id>/bootstrap/review/labels/`
  - YOLO label files (`.txt`): `class_id x_center y_center width height` (normalized 0..1)
- `artifacts/models/<run-id>/bootstrap/review/predictions.jsonl`
  - one JSON object per detection event
- `artifacts/models/<run-id>/bootstrap/review/classes.txt`
  - class prompt list used for this run
- `artifacts/models/<run-id>/bootstrap/review/summary.json`
  - run-level summary counters and paths

No XML files are generated in bootstrap output.

## Dataset Build and Validation

### Prepare from XML annotations

```bash
./cds dataset prepare --config config/dataset.yaml
```

This pipeline:
- converts Pascal VOC XML to YOLO txt labels
- generates deterministic or time-aware splits
- validates dataset health (bounds, duplicates, empties, class IDs)
- writes reports under `dataset/reports/`

### Validate existing YOLO dataset

```bash
./cds dataset validate --dataset-root dataset
```

Validation checks include:
- exact class ID range mapping to canonical class list
- out-of-bounds boxes
- empty labels
- duplicate label rows
- missing label/image pairs

## XML Formats in This Project

XML is used as an input annotation format for conversion into YOLO labels.

References:
- Pascal VOC project reference: [PASCAL VOC](https://host.robots.ox.ac.uk/pascal/VOC/)
- Pascal VOC XML format overview: [Pascal VOC XML](https://roboflow.com/formats/pascal-voc-xml)

Who creates XML files in this project:
- Legacy runtime path (`rtsp-object-ident.py`) can write XML annotations when using `--writeXmlOnly` or image write modes.
- External annotation tools (for example LabelImg) can produce Pascal VOC XML used by `cds dataset prepare`.

## Fine-Tune Training

Primary command:

```bash
./cds train --config config/train.yaml
```

Useful overrides:

```bash
./cds train \
  --config config/train.yaml \
  --model yolov8m.pt \
  --epochs 120 \
  --batch 8 \
  --device cuda \
  --dataset dataset/data.yaml \
  --output-dir artifacts/models
```

Expected artifact layout:
- `artifacts/models/<run-id>/checkpoints/best.pt`
- `artifacts/models/<run-id>/exports/`
- `artifacts/models/<run-id>/reports/train_summary.json`
- `artifacts/models/<run-id>/reports/export_report.json` (when export enabled)

## Evaluation and Promotion Gating

```bash
./cds evaluate --config config/eval.yaml
```

Report includes:
- precision/recall/mAP
- per-class metrics
- confusion matrix path (when generated)
- optional day/night subgroup metrics (if configured)
- optional runtime latency file inclusion

Promotion gate:
- model is a production candidate only if every required class meets configured recall threshold.

## Export Pipeline

```bash
./cds export --config config/export.yaml --targets onnx,coreml,tensorrt,rknn
```

Or export every supported target on host:

```bash
./cds export --config config/export.yaml --targets all
```

Unsupported targets are skipped with explicit warnings in `export_report.json`.

## Toolchain Preflight

Runtime preflight:

```bash
./cds doctor --target runtime
```

Training/conversion preflight:

```bash
./cds doctor --target training
```

Combined:

```bash
./cds doctor --target all
```

## Active Learning Loop (How To Use)

Active learning helps you recover from false positives/false negatives and retrain incrementally.

### Step 1: Capture runtime events

Run inference with event output enabled and keep JSONL events:
- each event includes source, label, confidence, bbox, frame id

### Step 2: Queue uncertain detections

```bash
./cds train active-learning queue \
  --events artifacts/runtime/events.jsonl \
  --output artifacts/active-learning/queue.jsonl \
  --min-conf 0.30 --max-conf 0.70
```

Optional reviewed truth input for stronger FP/FN candidate queueing:

```bash
./cds train active-learning queue \
  --events artifacts/runtime/events.jsonl \
  --truth artifacts/runtime/reviewed_truth.jsonl \
  --output artifacts/active-learning/queue.jsonl
```

### Step 3: Review and relabel

Review queued frames and update labels using your annotation workflow.

### Step 4: Merge reviewed items into dataset

```bash
./cds train active-learning merge \
  --queue artifacts/active-learning/queue.jsonl \
  --source-images /path/to/camera_data \
  --dataset dataset \
  --split train
```

### Step 5: Retrain and re-evaluate

```bash
./cds train --config config/train.yaml
./cds evaluate --config config/eval.yaml
```
