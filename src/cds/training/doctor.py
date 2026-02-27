from __future__ import annotations

import platform
import shutil
from typing import Any

from cds.training.export import available_export_targets


def _check_module(name: str, hint: str) -> dict[str, Any]:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "installed")
        return {"name": name, "ok": True, "version": str(version), "hint": ""}
    except Exception:
        return {"name": name, "ok": False, "version": None, "hint": hint}


def _check_onnxruntime() -> dict[str, Any]:
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        cuda = "CUDAExecutionProvider" in providers
        return {
            "name": "onnxruntime",
            "ok": True,
            "version": getattr(ort, "__version__", "installed"),
            "providers": providers,
            "cuda_provider": cuda,
            "hint": "",
        }
    except Exception:
        return {
            "name": "onnxruntime",
            "ok": False,
            "version": None,
            "providers": [],
            "cuda_provider": False,
            "hint": "Install onnxruntime (CPU) or onnxruntime-gpu (CUDA hosts)",
        }


def _check_ffmpeg() -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    return {
        "name": "ffmpeg",
        "ok": ffmpeg is not None,
        "path": ffmpeg,
        "hint": "Install ffmpeg system package" if ffmpeg is None else "",
    }


def _check_tensorrt_toolchain() -> dict[str, Any]:
    trtexec = shutil.which("trtexec")
    try:
        import tensorrt

        ver = getattr(tensorrt, "__version__", "installed")
        ok = trtexec is not None
        hint = "Install NVIDIA TensorRT runtime and ensure trtexec is on PATH"
        return {
            "name": "tensorrt",
            "ok": ok,
            "version": ver,
            "trtexec": trtexec,
            "hint": "" if ok else hint,
        }
    except Exception:
        return {
            "name": "tensorrt",
            "ok": False,
            "version": None,
            "trtexec": trtexec,
            "hint": "Install TensorRT Python package and trtexec utility",
        }


def _check_setuptools_pkg_resources() -> dict[str, Any]:
    try:
        import setuptools
        import pkg_resources  # noqa: F401

        version = getattr(setuptools, "__version__", "installed")
        return {
            "name": "setuptools",
            "ok": True,
            "version": str(version),
            "hint": "",
        }
    except Exception as exc:
        return {
            "name": "setuptools",
            "ok": False,
            "version": None,
            "error": str(exc),
            "hint": (
                "Install setuptools (provides pkg_resources). "
                "On Python 3.12 venvs this may be missing by default: "
                "python3 -m pip install setuptools"
            ),
        }


def _check_rknn_toolkit2() -> dict[str, Any]:
    try:
        import pkg_resources  # noqa: F401
    except Exception as exc:
        return {
            "name": "rknn_toolkit2",
            "ok": False,
            "version": None,
            "error": str(exc),
            "hint": (
                "Install setuptools first, then re-run the check. "
                "rknn-toolkit2 still imports pkg_resources."
            ),
        }

    try:
        import rknn
        from rknn.api import RKNN  # noqa: F401

        version = getattr(rknn, "__version__", "installed")
        return {
            "name": "rknn_toolkit2",
            "ok": True,
            "version": str(version),
            "hint": "",
        }
    except Exception as exc:
        return {
            "name": "rknn_toolkit2",
            "ok": False,
            "version": None,
            "error": str(exc),
            "hint": (
                "Install rknn-toolkit2 on the conversion host. "
                "rknnlite alone is runtime-only and cannot export .rknn models."
            ),
        }


def _check_rknnlite_runtime() -> dict[str, Any]:
    try:
        import rknnlite

        version = getattr(rknnlite, "__version__", "installed")
        return {
            "name": "rknnlite",
            "ok": True,
            "version": str(version),
            "hint": "",
        }
    except Exception as exc:
        return {
            "name": "rknnlite",
            "ok": False,
            "version": None,
            "error": str(exc),
            "hint": (
                "Install rknn-toolkit-lite2 on the deployment host "
                "to run exported .rknn models."
            ),
        }


def run_training_doctor() -> dict[str, Any]:
    checks = [
        _check_module("ultralytics", "pip install ultralytics"),
        _check_module(
            "clip",
            "pip install 'clip @ git+https://github.com/ultralytics/CLIP.git'",
        ),
        _check_module("onnx", "pip install onnx"),
        _check_onnxruntime(),
        _check_module(
            "coremltools",
            "pip install coremltools (typically used on macOS)",
        ),
        _check_tensorrt_toolchain(),
        _check_setuptools_pkg_resources(),
        _check_rknn_toolkit2(),
        _check_rknnlite_runtime(),
        _check_ffmpeg(),
    ]

    export_targets = available_export_targets()

    return {
        "target": "training",
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "checks": checks,
        "export_targets": export_targets,
        "notes": [
            "Linux+NVIDIA and macOS+MPS are primary training targets.",
            "Rockchip training is optional; inference/export conversion can run on separate hosts.",
            "Unsupported export targets should be skipped by cds export with warning.",
        ],
    }
