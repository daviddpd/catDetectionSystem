# CoreML Conversion Runbook

## Preconditions
- macOS host
- `ultralytics`, `onnx`, and `coremltools` installed
- trained checkpoint (`best.pt`)

If your source model is Darknet (`.cfg` + `.weights`) instead of `.pt`, first follow:
- `docs/runbooks/darknet-legacy-conversion.md`

## Command

```bash
./cds export --model artifacts/models/<run-id>/checkpoints/best.pt --targets coreml --output-dir artifacts/models/<run-id>
```

## Output
- `artifacts/models/<run-id>/exports/*.mlpackage` (or `.mlmodel` depending on runtime)
- `artifacts/models/<run-id>/reports/export_report.json`

## Naming Convention
- source checkpoint: `checkpoints/best.pt`
- exported coreml artifact: `exports/<model-stem>.mlpackage`

## Troubleshooting
- Missing `coremltools`: install with `pip install coremltools`
- Unsupported ops: try smaller model variant or export ONNX first and inspect ops
- Memory pressure: reduce `imgsz` during export
