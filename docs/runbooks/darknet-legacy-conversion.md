# Darknet Legacy Conversion Runbook

Use this runbook when your current runtime model is OpenCV/Darknet:

- `--cfg-path ./yolo/cfg/yolov-tiny-custom-416v6-64.cfg`
- `--weights-path ./yolo/weights/yolov-tiny-custom-416v6-64_final.weights`

## Scope and limitation

- `cds export` currently supports source models in `.pt` or `.onnx`.
- direct `.cfg` + `.weights` export to every target is not currently implemented in `cds export`.
- ONNX -> PyTorch checkpoint (`.pt`) is not a supported round-trip path in this project.

## Step 1: Run current Darknet model (no conversion)

```bash
./cds detect \
  --backend opencv-darknet \
  --cfg-path ./yolo/cfg/yolov-tiny-custom-416v6-64.cfg \
  --weights-path ./yolo/weights/yolov-tiny-custom-416v6-64_final.weights \
  --labels-path ./yolo/cfg/custom-names-v4.txt \
  --uri /path/to/video_or_rtsp
```

## Step 2: Create ONNX bridge artifact from Darknet model

One example external converter is [darknet-onnx](https://github.com/daviddpd/darknet-onnx).

Example flow:

```bash
export CDS_REPO=/path/to/catDetectionSystem
git clone https://github.com/daviddpd/darknet-onnx.git
cd darknet-onnx
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m darknetonnx.darknet \
  --cfg "$CDS_REPO/yolo/cfg/yolov-tiny-custom-416v6-64.cfg" \
  --weight "$CDS_REPO/yolo/weights/yolov-tiny-custom-416v6-64_final.weights"
```

This produces an ONNX file (typically `model.onnx` in the working directory). Move it into your CDS artifacts tree:

```bash
mkdir -p "$CDS_REPO/artifacts/models/darknet-bridge/exports"
mv model.onnx "$CDS_REPO/artifacts/models/darknet-bridge/exports/yolov-tiny-custom-416v6-64.onnx"
```

## Step 3: Export from bridge ONNX with CDS

ONNX + RKNN bundle:

```bash
cd "$CDS_REPO"
./cds export \
  --model artifacts/models/darknet-bridge/exports/yolov-tiny-custom-416v6-64.onnx \
  --targets onnx,rknn \
  --output-dir artifacts/models/darknet-bridge
```

Result:
- ONNX artifact copied into `artifacts/models/darknet-bridge/exports/`
- RKNN conversion bundle generated in `artifacts/models/darknet-bridge/rknn/`

## Full multi-target export (`onnx,coreml,tensorrt,rknn`)

For the full target set in the CDS pipeline, use a `.pt` checkpoint source:

```bash
./cds export \
  --model artifacts/models/<run-id>/checkpoints/best.pt \
  --targets all \
  --output-dir artifacts/models/<run-id>
```

This is the supported path for producing all Stage 2 export targets from one command.
