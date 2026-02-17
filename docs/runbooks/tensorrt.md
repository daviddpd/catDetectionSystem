# TensorRT Conversion Runbook

## Preconditions
- Linux NVIDIA host
- CUDA + TensorRT installed
- `trtexec` available on `PATH`
- `ultralytics`, `onnx`, `tensorrt` installed

If your source model is Darknet (`.cfg` + `.weights`) instead of `.pt`, first follow:
- `docs/runbooks/darknet-legacy-conversion.md`

## Command

```bash
./cds export --model artifacts/models/<run-id>/checkpoints/best.pt --targets tensorrt --output-dir artifacts/models/<run-id>
```

## Output
- `artifacts/models/<run-id>/exports/*.engine`
- `artifacts/models/<run-id>/reports/export_report.json`

## Naming Convention
- TensorRT engine is kept in `exports/` and recorded in export report.

## Troubleshooting
- `trtexec` not found: install TensorRT system components
- Unsupported precision/kernel: rerun without half precision
- Engine build fails: export ONNX first and test with `trtexec --onnx=<file>`
