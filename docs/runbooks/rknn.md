# RKNN Conversion Runbook

Stage 2 generates RKNN conversion bundles even when conversion cannot run on the current host.

If your source model is Darknet (`.cfg` + `.weights`), generate a bridge ONNX first:
- `docs/runbooks/darknet-legacy-conversion.md`

## Supported Chip Families

Toolkit2 family:
- RK3588
- RK3576
- RK3566
- RK3568
- RK3562
- RV1103
- RV1106
- RV1103B
- RV1106B
- RV1126B
- RK2118

Legacy toolkit family:
- RK1808
- RV1109
- RV1126
- RK3399Pro

## Generate Conversion Bundle

```bash
./cds export --model artifacts/models/<run-id>/checkpoints/best.pt --targets onnx,rknn --output-dir artifacts/models/<run-id>
```

Bundle output:
- `artifacts/models/<run-id>/rknn/convert_toolkit2.py`
- `artifacts/models/<run-id>/rknn/convert_toolkit2_vendor.py`
- `artifacts/models/<run-id>/rknn/convert_legacy.py`
- `artifacts/models/<run-id>/rknn/calibration.txt`
- `artifacts/models/<run-id>/rknn/make_calibration_txt.py`
- `artifacts/models/<run-id>/rknn/smoke_test_rknn.py`
- `artifacts/models/<run-id>/rknn/run_vendor_quant_smoke.sh`
- `artifacts/models/<run-id>/rknn/chip_families.txt`

## Calibration / Quantization Inputs

Prepare a calibration file containing representative image paths:
- one absolute path per line
- 100-1000 images recommended for stable quantization
- include day/night and weather variance

The generated conversion scripts set explicit preprocessing:
- `mean_values=[[0, 0, 0]]`
- `std_values=[[255, 255, 255]]`

That matches CDS runtime preprocessing (`RGB / 255.0`). Older generated bundles omitted this and can produce quantized RKNN models with live boxes but dead class scores (`cls_max=0.0`).

The generated scripts now look for `calibration.txt` in the same directory as the script, not the current shell directory.

Generate it from a directory of images:

```bash
python3 artifacts/models/<run-id>/rknn/make_calibration_txt.py \
  /path/to/calibration-images \
  --output artifacts/models/<run-id>/rknn/calibration.txt \
  --limit 200
```

Model-assisted selection is now supported as a first-class workflow. This is useful when you want to build `calibration.txt` on a faster host (for example macOS with CoreML) and keep only strong positives from your own source images:

```bash
python3 artifacts/models/<run-id>/rknn/make_calibration_txt.py \
  /path/to/calibration-images \
  --output artifacts/models/<run-id>/rknn/calibration.txt \
  --model-path artifacts/models/<run-id>/exports/best.mlpackage \
  --labels-path dataset/classes.txt \
  --backend auto \
  --imgsz 640 \
  --min-confidence 0.90 \
  --coverage-per-label 1 \
  --limit 1000
```

Notes:
- `--model-path` enables headless scoring with the CDS detector stack. `--backend auto` will choose the best supported backend on that host (for example CoreML on Apple Silicon for `.mlpackage` artifacts).
- `--use-bundle-model` is a convenience flag that auto-discovers `best.mlpackage`, `best.pt`, or `best.onnx` under the sibling `exports/` directory.
- `--coverage-per-label 1` biases selection toward at least one strong image per detected label before filling the remaining slots by score.
- Omit `--model-path` (and `--use-bundle-model`) to keep the original random-sampling behavior.

You can also write it manually:
- one absolute image path per line
- blank lines and `#` comment lines are ignored

If you want a quick functional conversion without quantization, edit the generated conversion script and set `DO_QUANTIZATION = False`.

Use that as a sanity check only. Confidence thresholds are backend- and artifact-specific, so a non-quantized RKNN build may need a different `--confidence` than the CoreML or PyTorch export of the same training run.

## Artifact Naming Convention

Toolkit2 output:
- `model.toolkit2.rknn`

Legacy output:
- `model.legacy.rknn`

## Conversion Execution

Run on a properly provisioned conversion host:

```bash
python3 -m pip install --force-reinstall 'setuptools<82'
python3 artifacts/models/<run-id>/rknn/convert_toolkit2.py
python3 artifacts/models/<run-id>/rknn/convert_legacy.py
```

For vendor-style comparison testing, use the CLI-driven Toolkit2 wrapper instead:

```bash
python3 artifacts/models/<run-id>/rknn/convert_toolkit2_vendor.py \
  --onnx artifacts/models/<run-id>/checkpoints/<model>.onnx \
  --output artifacts/models/<run-id>/rknn/model.toolkit2.vendor.rknn \
  --calibration artifacts/models/<run-id>/rknn/calibration.txt
```

That wrapper prints the expected runtime contract explicitly:
- `NHWC`
- `uint8`
- batched `4D` input
- raw `RGB` image bytes

One-shot build + smoke test on the RKNN host:

```bash
./artifacts/models/<run-id>/rknn/run_vendor_quant_smoke.sh /path/to/test-image.jpg
```

Standalone smoke test only (use an existing `.rknn`):

```bash
python3 artifacts/models/<run-id>/rknn/smoke_test_rknn.py \
  --image /path/to/test-image.jpg \
  --rknn-model artifacts/models/<run-id>/rknn/model.toolkit2.vendor.rknn \
  --onnx-model artifacts/models/<run-id>/checkpoints/<model>.onnx
```

The smoke test prints:
- RKNN input shape/type
- output tensor shapes
- per-output attribute maxima
- YOLO-style object/class maxima (raw and sigmoid) when it recognizes a detect head

## Troubleshooting
- Missing RKNN package: install toolkit package matching chip family
- `ModuleNotFoundError: No module named 'pkg_resources'`: `setuptools 82.0.0` removed `pkg_resources`; pin below 82 in that venv (`python3 -m pip install --force-reinstall 'setuptools<82'`)
- `rknnlite` installed but conversion still fails: `rknnlite` is runtime-only; ONNX -> `.rknn` conversion requires `rknn-toolkit2`
- Build errors from unsupported ONNX ops: simplify model, re-export ONNX, or use a smaller checkpoint
- Quantization instability: improve calibration coverage and image quality
- Quantized model runs but reports boxes with zero class scores: regenerate the RKNN bundle with the latest `./cds export` and rebuild. Older templates omitted explicit `mean/std` preprocessing in `rknn.config(...)`, which can collapse class channels during quantization.
- Non-quantized model detects but seems noisy: that is a threshold calibration issue, not a decode bug by itself. Re-tune `--confidence` for the RKNN artifact instead of copying the CoreML threshold directly.
- Runtime mismatch: ensure generated RKNN artifact matches NPU generation and runtime version
