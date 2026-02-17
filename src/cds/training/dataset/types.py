from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetRecord:
    image_path: Path
    label_path: Path
    source_id: str
    timestamp: str | None


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    image_path: str
    label_path: str | None = None
