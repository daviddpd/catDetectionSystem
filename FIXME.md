# FIXME

## RKNN Quantized Export Investigation

- Status: non-quantized `.rknn` detects correctly on RK3588; quantized `.rknn` still returns live box channels but zeroed class channels (`cls_max=0.0`) even when CDS feeds the runtime in the accepted format (`NHWC`, `uint8`, 4D batched, `640x640x3`).
- Next step: compare against a known-good quantized RKNN reference model and/or a minimal proof-of-concept ONNX -> RKNN conversion to determine whether the failure is specific to the current Ultralytics export path, the calibration set, or RKNN Toolkit2 quantization itself.
- Follow-up: add a standalone RKNN smoke-test helper that runs one image through a freshly built `.rknn` artifact and prints `attr_max` / `cls_max` so broken quantized exports are caught immediately after conversion.

## ONNX Version / Opset Drift Between Hosts

- macOS and Rockchip/Linux currently need different ONNX package versions because of dependency pressure from `coremltools` vs `rknn-toolkit2`.
- Observed failure: `Unsupport onnx opset 22, need <= 19`.
- Follow-up: pin/document host-specific ONNX version ranges and export opset settings so the generated ONNX bridge artifact is accepted by both CoreML and RKNN conversion workflows, or split the export paths explicitly by host target.

## Path Portability Across Hosts and Mounts

- Current workflows assume absolute paths inside generated files (`calibration.txt`, manifests, export scripts).
- This breaks when the same artifact bundle is moved between:
  - Linux hosts using NFS mount paths
  - macOS hosts using SMB mount paths
  - different home-directory roots
- Temporary workaround: local symlink shims.
- Follow-up: introduce a path-rewrite / path-map layer (or a repo-relative manifest mode) so bundles can be generated on one host and consumed on another without manual symlink surgery.
