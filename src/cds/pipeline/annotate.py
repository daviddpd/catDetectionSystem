from __future__ import annotations

import cv2

from cds.types import Detection


def _color_for_label(label: str) -> tuple[int, int, int]:
    seed = abs(hash(label))
    b = 50 + (seed % 180)
    g = 50 + ((seed >> 8) % 180)
    r = 50 + ((seed >> 16) % 180)
    return int(b), int(g), int(r)


def draw_overlays(
    frame,
    detections: list[Detection],
    backend: str,
    fps_infer: float,
) -> None:
    top_y = 28
    cv2.putText(
        frame,
        f"backend={backend} fps={fps_infer:.2f}",
        (12, top_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for detection in detections:
        color = _color_for_label(detection.label)
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            color,
            2,
        )
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (detection.x1, max(20, detection.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            2,
            cv2.LINE_AA,
        )
