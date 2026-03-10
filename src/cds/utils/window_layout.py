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
    return left, right


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


def place_windows_side_by_side(left_window: str, right_window: str) -> bool:
    try:
        import cv2  # noqa: PLC0415
    except Exception:
        return False

    screen_w, screen_h = detect_screen_size()
    left_rect, right_rect = compute_side_by_side_rects(screen_w, screen_h)

    try:
        cv2.resizeWindow(left_window, left_rect.width, left_rect.height)
        cv2.moveWindow(left_window, left_rect.x, left_rect.y)
        cv2.resizeWindow(right_window, right_rect.width, right_rect.height)
        cv2.moveWindow(right_window, right_rect.x, right_rect.y)
        return True
    except Exception:
        return False
