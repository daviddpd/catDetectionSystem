from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelSpec:
    name: str
    model_path: str | None = None
    cfg_path: str | None = None
    weights_path: str | None = None
    labels_path: str | None = None
    confidence: float = 0.5
    nms: float = 0.5
    imgsz: int = 640
    class_filter: set[str] = field(default_factory=set)

    def read_labels(self) -> list[str]:
        if not self.labels_path:
            return []
        path = Path(self.labels_path)
        if not path.exists():
            return []
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
