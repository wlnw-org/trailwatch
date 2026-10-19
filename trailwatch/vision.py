"""OpenCV 5 perception tools for camera-trap video.

Each function is a small, deterministic tool the agent can call. Every result
carries the measurements that justify it, so the agent's later decisions can be
traced back to visual evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    image: np.ndarray = field(repr=False)


@dataclass
class MotionEvent:
    start: int
    end: int
    peak_frame: int
    peak_area: float
    box: tuple[int, int, int, int]  # x, y, w, h on the peak frame
    boxes: dict[int, tuple[int, int, int, int]] = field(default_factory=dict, repr=False)

    def box_for(self, frame_index: int) -> tuple[int, int, int, int]:
        return self.boxes.get(frame_index, self.box)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("boxes", None)
        return d


@dataclass
class Quality:
    brightness: float      # mean grey level, 0 to 255
    sharpness: float       # variance of the Laplacian
    is_night_ir: bool      # near-zero colour saturation, typical of infrared frames
    usable: bool

    def to_dict(self) -> dict:
        return asdict(self)


def read_frames(path: str, step: int = 1, max_frames: int = 3000) -> tuple[list[Frame], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames: list[Frame] = []
    i = 0
    while len(frames) < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(Frame(i, img))
        i += 1
    cap.release()
    return frames, fps


def detect_motion(frames: list[Frame], min_area_frac: float = 0.002, var_threshold: float = 32.0,
                  gap: int = 5) -> list[MotionEvent]:
    """Background subtraction (MOG2) plus morphology; groups moving frames into events."""
    if not frames:
        return []
    h, w = frames[0].image.shape[:2]
    min_area = min_area_frac * h * w
    sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=var_threshold, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hits: list[tuple[int, float, tuple[int, int, int, int]]] = []
    for n, f in enumerate(frames):
        mask = sub.apply(f.image)
        if n < 3:  # let the model settle
            continue
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]  # drop shadows (value 127)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in contours if cv2.contourArea(c) >= min_area]
        if big:
            c = max(big, key=cv2.contourArea)
            hits.append((f.index, float(cv2.contourArea(c)), cv2.boundingRect(c)))
    events: list[MotionEvent] = []
    for idx, area, box in hits:
        if events and idx - events[-1].end <= gap:
            e = events[-1]
            e.end = idx
            if area > e.peak_area:
                e.peak_area, e.peak_frame, e.box = area, idx, box
        else:
            events.append(MotionEvent(idx, idx, idx, area, box))
        events[-1].boxes[idx] = box
    return events


def assess_quality(img: np.ndarray, min_brightness: float = 40.0, min_sharpness: float = 60.0) -> Quality:
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(grey.mean())
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    sat = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean())
    return Quality(round(brightness, 1), round(sharpness, 1), sat < 8.0,
                   brightness >= min_brightness and sharpness >= min_sharpness)


def enhance(img: np.ndarray, target_mean: float = 110.0) -> np.ndarray:
    """Low-light and haze recovery.

    1. Adaptive gamma: lift dark frames so the mean grey level approaches target_mean.
    2. CLAHE on the L channel for local contrast.
    3. A mild unsharp mask to restore edges.
    """
    grey_mean = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    out = img
    if 1.0 < grey_mean < target_mean:
        gamma = np.log(target_mean / 255.0) / np.log(grey_mean / 255.0)
        lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 2.0)
    return cv2.addWeighted(out, 1.6, blur, -0.6, 0)


def crop(img: np.ndarray, box: tuple[int, int, int, int], pad: float = 0.25) -> np.ndarray:
    x, y, w, h = box
    H, W = img.shape[:2]
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + w + px), min(H, y + h + py)
    return img[y0:y1, x0:x1].copy()


def best_frame(frames: list[Frame], event: MotionEvent, avoid: set[int] | None = None) -> Frame:
    """The sharpest frame inside an event, skipping frames already tried."""
    avoid = avoid or set()
    cands = [f for f in frames if f.index in event.boxes and f.index not in avoid]
    if not cands:
        cands = [f for f in frames if f.index == event.peak_frame]
    return max(cands, key=lambda f: assess_quality(f.image).sharpness)
