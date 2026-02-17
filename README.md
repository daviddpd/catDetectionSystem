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
- `docs/runbooks/coreml.md`
- `docs/runbooks/tensorrt.md`
- `docs/runbooks/rknn.md`

