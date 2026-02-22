from __future__ import annotations

import hashlib

from cds.types import Detection


def _color_for_label(label: str) -> tuple[int, int, int]:
    # Use a deterministic hash so each class color is stable across process runs.
    digest = hashlib.sha1(label.encode("utf-8")).digest()
    b = 50 + (digest[0] % 180)
    g = 50 + (digest[1] % 180)
    r = 50 + (digest[2] % 180)
    return int(b), int(g), int(r)


def draw_overlays(
    frame,
    detections: list[Detection],
    backend: str,
    fps_infer: float,
) -> None:
    import cv2


    # 1200 x 100, 1175, 88, 12, 36

    reference_x = 12
    reference_y = 36
    status_text_x = 1300
    status_text_y = 36
    cv2.putText(
        frame,
        f"backend={backend} fps={fps_infer:.2f}",
        (status_text_x, status_text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
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
            3,
        )
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (detection.x1, max(20, detection.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            color,
            3,
            cv2.LINE_AA,
        )
