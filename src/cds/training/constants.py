from __future__ import annotations

DEFAULT_BOOTSTRAP_MODEL = "yolov8s-worldv2.pt"
DEFAULT_TRAIN_MODEL = "yolov8s.pt"

TOOLKIT2_CHIPS: list[str] = [
    "RK3588",
    "RK3576",
    "RK3566",
    "RK3568",
    "RK3562",
    "RV1103",
    "RV1106",
    "RV1103B",
    "RV1106B",
    "RV1126B",
    "RK2118",
]

LEGACY_RKNN_CHIPS: list[str] = [
    "RK1808",
    "RV1109",
    "RV1126",
    "RK3399Pro",
]
