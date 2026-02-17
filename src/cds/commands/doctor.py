from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

from cds.config import load_runtime_config
from cds.detector import select_backend
from cds.detector.models import ModelSpec
from cds.io.ingest import probe_decoder_path, select_ingest_backend


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return None


def _build_model_spec(config) -> ModelSpec:
    return ModelSpec(
        name=config.model.name,
        model_path=config.model.path,
        cfg_path=config.model.cfg_path,
        weights_path=config.model.weights_path,
        labels_path=config.model.labels_path,
        confidence=config.model.confidence,
        nms=config.model.nms,
        imgsz=config.model.imgsz,
        class_filter=set(config.model.class_filter),
    )


def run_doctor(args: Any, repo_root: Path) -> int:
    config = load_runtime_config(repo_root=repo_root, config_path=args.config)

    decoder = probe_decoder_path()
    ingest_probe: dict[str, Any] = {
        "ok": False,
        "selected": None,
        "reason": None,
        "error": None,
    }
    try:
        ingest, ingest_reason = select_ingest_backend(config.ingest)
        ingest_probe = {
            "ok": True,
            "selected": ingest.name(),
            "reason": ingest_reason,
            "error": None,
        }
    except Exception as exc:
        ingest_probe["error"] = str(exc)

    backend_probe: dict[str, Any] = {
        "ok": False,
        "selected": None,
        "reason": None,
        "device": None,
        "error": None,
    }
    try:
        model_spec = _build_model_spec(config)
        selection = select_backend(model_spec, config.backend_policy)
        backend_probe = {
            "ok": True,
            "selected": selection.backend.name(),
            "reason": selection.reason,
            "device": selection.backend.device_info(),
            "error": None,
        }
    except Exception as exc:
        backend_probe["error"] = str(exc)

    report = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "modules": {
            "opencv": _module_version("cv2"),
            "ultralytics": _module_version("ultralytics"),
            "torch": _module_version("torch"),
            "av": _module_version("av"),
            "dynaconf": _module_version("dynaconf"),
            "prometheus_client": _module_version("prometheus_client"),
            "rknnlite": _module_version("rknnlite"),
        },
        "decoder": {
            "selected": decoder.selected_decoder,
            "reason": decoder.reason,
            "available": decoder.available,
        },
        "ingest": {
            "selected": ingest_probe["selected"],
            "reason": ingest_probe["reason"],
            "ok": ingest_probe["ok"],
            "error": ingest_probe["error"],
        },
        "backend": backend_probe,
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        print("cds doctor report")
        print(f"- platform: {report['platform']['system']} {report['platform']['machine']}")
        print(f"- python: {report['platform']['python']}")
        print(f"- decoder: {report['decoder']['selected']} ({report['decoder']['reason']})")
        if ingest_probe["ok"]:
            print(f"- ingest: {report['ingest']['selected']} ({report['ingest']['reason']})")
        else:
            print(f"- ingest: unavailable ({report['ingest']['error']})")
        if backend_probe["ok"]:
            print(
                "- backend: "
                f"{backend_probe['selected']} [{backend_probe['device']}] ({backend_probe['reason']})"
            )
        else:
            print(f"- backend: unavailable ({backend_probe['error']})")
        print("- modules:")
        for name, version in report["modules"].items():
            print(f"  - {name}: {version or 'not installed'}")

    return 0
