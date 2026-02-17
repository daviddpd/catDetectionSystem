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
- `artifacts/models/<run-id>/rknn/convert_legacy.py`
- `artifacts/models/<run-id>/rknn/chip_families.txt`

## Calibration / Quantization Inputs

Prepare a calibration file containing representative image paths:
- one absolute path per line
- 100-1000 images recommended for stable quantization
- include day/night and weather variance

Set this path in conversion scripts as `CALIBRATION_DATASET`.

## Artifact Naming Convention

Toolkit2 output:
- `model.toolkit2.rknn`

Legacy output:
- `model.legacy.rknn`

## Conversion Execution

Run on a properly provisioned conversion host:

```bash
python3 artifacts/models/<run-id>/rknn/convert_toolkit2.py
python3 artifacts/models/<run-id>/rknn/convert_legacy.py
```

## Troubleshooting
- Missing RKNN package: install toolkit package matching chip family
- Build errors from unsupported ONNX ops: simplify model, re-export ONNX, or use a smaller checkpoint
- Quantization instability: improve calibration coverage and image quality
- Runtime mismatch: ensure generated RKNN artifact matches NPU generation and runtime version
