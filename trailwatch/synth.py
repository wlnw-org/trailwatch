"""Synthetic camera-trap clips for tests and the public demo (no real animals or people)."""
from __future__ import annotations

import cv2
import numpy as np


def make_clip(path: str, kind: str = "elephant", frames: int = 60, size: tuple[int, int] = (480, 320),
              dark: bool = False, blur: int = 0, seed: int = 0) -> str:
    """Write a short MJPG/AVI clip of a coloured shape crossing a textured background.

    kind: "elephant" (red), "deer" (blue), "unknown" (grey) or "empty".
    """
    rng = np.random.default_rng(seed)
    w, h = size
    base = (rng.integers(40, 110, (h, w, 3))).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 3)
    colour = {"elephant": (40, 40, 200), "deer": (200, 60, 40), "unknown": (120, 120, 120)}.get(kind)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    for i in range(frames):
        img = base.copy()
        noise = rng.integers(-6, 7, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if colour and 10 <= i < frames - 5:
            x = int(20 + (w - 140) * (i - 10) / max(1, frames - 15))
            cv2.ellipse(img, (x + 60, h // 2), (60, 40), 0, 0, 360, colour, -1)
            cv2.circle(img, (x + 120, h // 2 - 25), 18, colour, -1)
            for k in range(0, 120, 12):  # texture so sharpness is measurable
                cv2.line(img, (x + k, h // 2 - 30), (x + k + 6, h // 2 + 30), (20, 20, 20), 1)
        if blur:
            img = cv2.GaussianBlur(img, (0, 0), blur)
        if dark:
            img = (img * 0.22).astype(np.uint8)
        vw.write(img)
    vw.release()
    return path
