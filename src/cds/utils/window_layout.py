from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowRect:
    x: int
    y: int
    width: int
    height: int


def compute_side_by_side_rects(
    screen_width: int,
    screen_height: int,
    *,
    margin: int = 24,
    gap: int = 16,
    right_scale: float = 1.0,
) -> tuple[WindowRect, WindowRect]:
    safe_w = max(800, int(screen_width))
    safe_h = max(600, int(screen_height))
    safe_margin = max(8, int(margin))
    safe_gap = max(0, int(gap))

    available_w = max(640, safe_w - (2 * safe_margin) - safe_gap)
    panel_w = max(480, available_w // 2)
    panel_h = max(360, int((safe_h - (2 * safe_margin)) * 0.90))

    y = safe_margin
    left_x = safe_margin
    right_x = safe_margin + panel_w + safe_gap

    left = WindowRect(x=left_x, y=y, width=panel_w, height=panel_h)
    right = WindowRect(x=right_x, y=y, width=panel_w, height=panel_h)
    safe_scale = max(0.05, min(1.0, float(right_scale)))
    if safe_scale < 0.999:
        scaled_w = max(240, int(right.width * safe_scale))
        scaled_h = max(180, int(right.height * safe_scale))
        scaled_y = right.y + max(0, (right.height - scaled_h) // 2)
        right = WindowRect(
            x=right.x,
            y=scaled_y,
            width=scaled_w,
            height=scaled_h,
        )
    return left, right


def compute_single_rect(
    screen_width: int,
    screen_height: int,
    *,
    margin: int = 24,
    scale: float = 1.0,
) -> WindowRect:
    left_rect, _ = compute_side_by_side_rects(
        screen_width,
        screen_height,
        margin=margin,
        gap=16,
    )
    safe_scale = max(0.10, min(1.0, float(scale)))
    if safe_scale >= 0.999:
        return left_rect

    scaled_w = max(320, int(left_rect.width * safe_scale))
    scaled_h = max(240, int(left_rect.height * safe_scale))
    scaled_x = left_rect.x + max(0, (left_rect.width - scaled_w) // 2)
    scaled_y = left_rect.y + max(0, (left_rect.height - scaled_h) // 2)
    return WindowRect(
        x=scaled_x,
        y=scaled_y,
        width=scaled_w,
        height=scaled_h,
    )


def detect_screen_size() -> tuple[int, int]:
    try:
        import tkinter  # noqa: PLC0415

        root = tkinter.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return 1920, 1080


def place_windows_side_by_side(
    left_window: str,
    right_window: str,
    *,
    right_window_scale: float = 1.0,
) -> bool:
    try:
        import cv2  # noqa: PLC0415
    except Exception:
        return False

    screen_w, screen_h = detect_screen_size()
    left_rect, right_rect = compute_side_by_side_rects(
        screen_w,
        screen_h,
        right_scale=right_window_scale,
    )

    try:
        cv2.resizeWindow(left_window, left_rect.width, left_rect.height)
        cv2.moveWindow(left_window, left_rect.x, left_rect.y)
        cv2.resizeWindow(right_window, right_rect.width, right_rect.height)
        cv2.moveWindow(right_window, right_rect.x, right_rect.y)
        return True
    except Exception:
        return False


def place_single_window(
    window_name: str,
    *,
    window_scale: float = 1.0,
) -> bool:
    try:
        import cv2  # noqa: PLC0415
    except Exception:
        return False

    screen_w, screen_h = detect_screen_size()
    rect = compute_single_rect(
        screen_w,
        screen_h,
        scale=window_scale,
    )
    try:
        cv2.resizeWindow(window_name, rect.width, rect.height)
        cv2.moveWindow(window_name, rect.x, rect.y)
        return True
    except Exception:
        return False
