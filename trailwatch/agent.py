"""Perception, decision and action loop for camera-trap triage.

The agent never acts on a fixed vision result. Each step looks at what the
OpenCV tools measured and chooses the next tool call:

  * no motion found          -> re-run detection with a lower threshold
  * frame too dark or blurry -> enhance it, then re-check quality
  * still unusable           -> try the next sharpest frame of the event
  * classifier unsure        -> try another frame or the enhanced crop
  * then decide an action    -> log, send to human review, or raise an alert
                                that waits for a person to approve it

Every step is written to a trace so judges and reviewers can see why each
decision was made.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2

from . import vision
from .classify import Classifier, Prediction, load_classifier

DEFAULT_POLICY = {
    "accept_score": 0.70,          # at or above: confident enough to log automatically
    "review_score": 0.30,          # below accept but above this: human review
    "max_frame_attempts": 3,
    "watchlist": {                 # species that trigger an alert, with the reason shown to people
        "Elephas maximus": "elephant near the station: possible crop-raiding or human-elephant conflict",
        "Panthera tigris": "tiger recorded",
        "Panthera pardus": "leopard recorded",
        "Crocodylus palustris": "mugger crocodile outside the enclosure zone",
    },
    "alerts_need_approval": True,  # a person confirms before any alert leaves the system
}


@dataclass
class Step:
    t: float
    tool: str
    inputs: dict
    observation: dict
    decision: str


@dataclass
class EventResult:
    event: dict
    frame_used: int | None
    quality: dict | None
    prediction: dict | None
    action: str
    reason: str
    attempts: int


@dataclass
class Run:
    video: str
    classifier: str
    fps: float
    frames_read: int
    events: list[EventResult] = field(default_factory=list)
    trace: list[Step] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class TriageAgent:
    def __init__(self, classifier: Classifier | None = None, policy: dict | None = None, out_dir: str | None = None):
        self.clf = classifier or load_classifier()
        self.policy = {**DEFAULT_POLICY, **(policy or {})}
        self.out_dir = Path(out_dir) if out_dir else None

    def _log(self, run: Run, tool: str, inputs: dict, observation: dict, decision: str) -> None:
        run.trace.append(Step(round(time.time(), 3), tool, inputs, observation, decision))

    def run(self, video_path: str) -> Run:
        frames, fps = vision.read_frames(video_path)
        run = Run(video_path, self.clf.name, fps, len(frames))
        self._log(run, "read_frames", {"path": video_path}, {"frames": len(frames), "fps": fps},
                  "detect motion")

        params = {"min_area_frac": 0.002, "var_threshold": 32.0}
        events = vision.detect_motion(frames, **params)
        self._log(run, "detect_motion", params, {"events": len(events)},
                  "triage each event" if events else "no motion: retry with a more sensitive threshold")
        if not events:
            params = {"min_area_frac": 0.0008, "var_threshold": 16.0}
            events = vision.detect_motion(frames, **params)
            self._log(run, "detect_motion", params, {"events": len(events)},
                      "triage each event" if events else "still nothing: mark clip as empty")

        for ev in events:
            run.events.append(self._triage(run, frames, ev))
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "trace.json").write_text(json.dumps(run.to_dict(), indent=2, default=str))
        return run

    def _triage(self, run: Run, frames: list[vision.Frame], ev: vision.MotionEvent) -> EventResult:
        p = self.policy
        tried: set[int] = set()
        best: tuple[Prediction, int, vision.Quality] | None = None
        attempts = 0
        while attempts < p["max_frame_attempts"]:
            attempts += 1
            f = vision.best_frame(frames, ev, avoid=tried)
            tried.add(f.index)
            img = f.image
            q = vision.assess_quality(img)
            if not q.usable:
                img = vision.enhance(img)
                q2 = vision.assess_quality(img)
                self._log(run, "enhance", {"frame": f.index, "before": q.to_dict()}, {"after": q2.to_dict()},
                          "use enhanced frame" if q2.usable else "enhanced but still poor: classify, then decide")
                q = q2
            c = vision.crop(img, ev.box_for(f.index))
            pred = self.clf.classify(c)
            self._log(run, "classify", {"frame": f.index, "crop": list(c.shape[:2])}, asdict(pred),
                      "accept" if pred.score >= p["accept_score"] else "unsure: try another frame")
            if best is None or pred.score > best[0].score:
                best = (pred, f.index, q)
                if self.out_dir:
                    self.out_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(self.out_dir / f"event{ev.start}_frame{f.index}.jpg"), c)
            if pred.score >= p["accept_score"]:
                break
            if len(tried) >= max(1, len(ev.boxes)):
                break

        pred, idx, q = best  # type: ignore[misc]
        action, reason = self._decide(pred, q)
        self._log(run, "decide", {"event": ev.to_dict()}, {"label": pred.label, "score": pred.score},
                  f"{action}: {reason}")
        return EventResult(ev.to_dict(), idx, q.to_dict(), asdict(pred), action, reason, attempts)

    def _decide(self, pred: Prediction, q: vision.Quality) -> tuple[str, str]:
        p = self.policy
        if pred.scientific in p["watchlist"]:
            if pred.score >= p["review_score"]:
                act = "alert_pending_approval" if p["alerts_need_approval"] else "alert"
                return act, p["watchlist"][pred.scientific]
            return "review", "possible watchlist species, low confidence"
        if pred.label in ("human", "vehicle"):
            return "discard", "person or vehicle: not kept, for privacy"
        if pred.label == "blank":
            return "discard", "empty frame"
        if pred.score >= p["accept_score"]:
            return "log", "confident identification"
        if pred.score >= p["review_score"]:
            return "review", "classifier unsure; a person should confirm"
        return "review", "poor image or unknown animal" + (" (night infrared)" if q.is_night_ir else "")
