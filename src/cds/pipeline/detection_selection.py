from __future__ import annotations

from cds.types import Detection


def select_primary_detection(detections: list[Detection]) -> list[Detection]:
    """Return a single best detection for downstream runtime processing."""
    if len(detections) <= 1:
        return detections

    best = max(
        detections,
        key=lambda det: (
            float(det.confidence),
            int(det.extra.get("area_pixels", det.width * det.height)),
        ),
    )
    return [best]
