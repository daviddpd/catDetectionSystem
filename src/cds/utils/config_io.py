from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config_file(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    suffix = p.suffix.lower()
    raw = p.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(raw)

    if suffix == ".toml":
        try:
            import tomllib
        except ImportError:  # pragma: no cover
            import tomli as tomllib  # type: ignore

        return tomllib.loads(raw)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "YAML config requires pyyaml. Install pyyaml or use TOML/JSON config."
            ) from exc
        loaded = yaml.safe_load(raw)
        return loaded if loaded else {}

    raise RuntimeError(f"Unsupported config extension: {suffix}")


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
