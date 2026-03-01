from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cds.config.models import BackendPolicyConfig
from cds.detector.backends.base import DetectorBackend
from cds.detector.errors import BackendUnavailable, ModelLoadError
from cds.detector.models.model_spec import ModelSpec


@dataclass
class BackendSelection:
    backend: DetectorBackend
    reason: str


def _has_torch() -> bool:
    try:
        import torch

        _ = torch.__version__
        return True
    except Exception:
        return False


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _torch_mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _ultralytics_available() -> bool:
    try:
        import ultralytics

        _ = ultralytics.__version__
        return True
    except Exception:
        return False


def _prime_ultralytics_for_rknn() -> None:
    """Mirror the auto-selection import order for RKNN.

    On some Rockchip hosts, importing top-level ultralytics before rknnlite
    avoids helper import failures inside the RKNN backend. If the import still
    fails, the RKNN backend will surface the precise postprocess error later.
    """
    try:
        import ultralytics

        _ = ultralytics.__version__
    except Exception:
        return


def _is_apple_silicon() -> bool:
    return platform.system().lower() == "darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }


def _is_linux() -> bool:
    return platform.system().lower() == "linux"


def _is_rockchip() -> bool:
    if not _is_linux():
        return False
    machine = platform.machine().lower()
    if any(key in machine for key in ("rk", "aarch64")):
        if shutil.which("rknn_server"):
            return True
        if any(
            path.exists()
            for path in (
                Path("/dev/rknpu"),
                Path("/usr/lib/librknnrt.so"),
                Path("/usr/lib64/librknnrt.so"),
            )
        ):
            return True
    return False


def _has_tensorrt() -> bool:
    try:
        import tensorrt

        _ = tensorrt.__version__
        return True
    except Exception:
        return False


def _new_ultralytics_backend(device: str) -> DetectorBackend:
    from cds.detector.backends.ultralytics_backend import UltralyticsBackend

    return UltralyticsBackend(device=device)


def _new_opencv_darknet_backend(device: str) -> DetectorBackend:
    from cds.detector.backends.opencv_darknet_backend import OpenCVDarknetBackend

    return OpenCVDarknetBackend(device=device)


def _new_rknn_backend() -> DetectorBackend:
    from cds.detector.backends.rknn_backend import RKNNBackend

    return RKNNBackend()


def _prefer_opencv_gpu_device() -> str:
    try:
        import cv2
    except Exception:
        return "cpu"

    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return "cuda"
    except Exception:
        pass

    try:
        if cv2.ocl.haveOpenCL():
            return "opencl"
    except Exception:
        pass

    return "cpu"


def _choose_with_fallback(
    candidates: list[tuple[Callable[[], DetectorBackend], str]],
    model_spec: ModelSpec,
) -> BackendSelection:
    errors: list[str] = []
    for factory, reason in candidates:
        backend: DetectorBackend | None = None
        try:
            backend = factory()
            backend.load(model_spec)
            return BackendSelection(backend=backend, reason=reason)
        except (BackendUnavailable, ModelLoadError, FileNotFoundError) as exc:
            backend_name = backend.name() if backend is not None else "unknown-backend"
            errors.append(f"{backend_name}: {exc}")
            continue
        except Exception as exc:
            backend_name = backend.name() if backend is not None else "unknown-backend"
            errors.append(f"{backend_name}: {exc}")
            continue
    raise RuntimeError(
        "No inference backend could be initialized. "
        + ("; ".join(errors) if errors else "No candidates evaluated.")
    )


def select_backend(
    model_spec: ModelSpec,
    policy: BackendPolicyConfig,
) -> BackendSelection:
    requested = policy.requested.lower().strip()

    if platform.system().lower() == "windows":
        raise RuntimeError("Windows is not supported in Stage 1 runtime.")

    if requested not in {"", "auto"}:
        if requested in {"ultralytics", "coreml", "mps", "cpu", "cuda", "tensorrt"}:
            device = "cpu"
            if requested == "cuda":
                device = "cuda"
            elif requested == "mps":
                device = "mps"
            elif requested == "coreml":
                device = "coreml"
            elif requested == "tensorrt":
                device = "cuda"
            return _choose_with_fallback(
                [
                    (
                        lambda: _new_ultralytics_backend(device=device),
                        f"Requested backend '{requested}' using ultralytics({device})",
                    )
                ],
                model_spec,
            )

        if requested in {"opencv", "opencv-darknet", "darknet"}:
            device = _prefer_opencv_gpu_device()
            return _choose_with_fallback(
                [
                    (
                        lambda: _new_opencv_darknet_backend(device=device),
                        f"Requested OpenCV/Darknet backend with device={device}",
                    )
                ],
                model_spec,
            )

        if requested == "rknn":
            _prime_ultralytics_for_rknn()
            return _choose_with_fallback(
                [(lambda: _new_rknn_backend(), "Requested RKNN backend")],
                model_spec,
            )

        raise RuntimeError(f"Unsupported backend requested: {requested}")

    # Auto selection policy:
    # 1) macOS Apple Silicon: coreml -> mps -> cpu
    # 2) Linux + NVIDIA: TensorRT (.engine) -> CUDA -> OpenCV GPU -> CPU
    # 3) Linux + Rockchip: RKNN -> CPU
    # 4) fallback: CPU
    candidates: list[tuple[Callable[[], DetectorBackend], str]] = []

    if _is_apple_silicon():
        if _ultralytics_available() and model_spec.model_path and model_spec.model_path.endswith(
            (".mlpackage", ".mlmodel")
        ):
            candidates.append(
                (
                    lambda: _new_ultralytics_backend(device="coreml"),
                    "macOS Apple Silicon policy: selected coreml from model artifact",
                )
            )
        if _ultralytics_available() and _torch_mps_available():
            candidates.append(
                (
                    lambda: _new_ultralytics_backend(device="mps"),
                    "macOS Apple Silicon policy: coreml unavailable, selected mps",
                )
            )
        if _ultralytics_available():
            candidates.append(
                (
                    lambda: _new_ultralytics_backend(device="cpu"),
                    "macOS Apple Silicon policy: using CPU ultralytics fallback",
                )
            )

    elif _is_linux() and _is_rockchip() and policy.allow_rknn:
        candidates.append(
            (
                lambda: _new_rknn_backend(),
                "Linux Rockchip policy: RKNN selected",
            )
        )
        if _ultralytics_available():
            candidates.append(
                (
                    lambda: _new_ultralytics_backend(device="cpu"),
                    "Linux Rockchip policy: RKNN unavailable, CPU ultralytics fallback",
                )
            )

    elif _is_linux():
        has_nvidia = shutil.which("nvidia-smi") is not None
        if (
            has_nvidia
            and policy.allow_tensorrt
            and model_spec.model_path
            and model_spec.model_path.endswith(".engine")
            and _has_tensorrt()
            and _ultralytics_available()
        ):
            candidates.append(
                (
                    lambda: _new_ultralytics_backend(device="cuda"),
                    "Linux NVIDIA policy: TensorRT engine artifact selected",
                )
            )

        if has_nvidia and _ultralytics_available() and _torch_cuda_available():
            candidates.append(
                (
                    lambda: _new_ultralytics_backend(device="cuda"),
                    "Linux NVIDIA policy: CUDA selected",
                )
            )

        if _ultralytics_available() and _has_torch():
            candidates.append(
                (
                    lambda: _new_ultralytics_backend(device="cpu"),
                    "Linux policy: ultralytics CPU fallback",
                )
            )

        opencv_device = _prefer_opencv_gpu_device()
        if policy.allow_darknet_fallback:
            candidates.append(
                (
                    lambda: _new_opencv_darknet_backend(device=opencv_device),
                    f"Linux policy: OpenCV/Darknet fallback with {opencv_device}",
                )
            )

    else:
        if _ultralytics_available():
            candidates.append(
                (
                    lambda: _new_ultralytics_backend(device="cpu"),
                    "Default policy: ultralytics CPU",
                )
            )

    if policy.allow_darknet_fallback:
        candidates.append(
            (
                lambda: _new_opencv_darknet_backend(device="cpu"),
                "Final fallback: OpenCV/Darknet CPU",
            )
        )

    return _choose_with_fallback(candidates, model_spec)
