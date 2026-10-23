import os

import cv2
from fastapi.testclient import TestClient

from trailwatch import vision
from trailwatch.agent import TriageAgent
from trailwatch.api import app
from trailwatch.classify import StubClassifier
from trailwatch.synth import make_clip


def test_opencv_5():
    assert cv2.__version__.startswith("5.")


def test_detects_motion(tmp_path):
    p = make_clip(str(tmp_path / "e.avi"), "elephant")
    frames, _ = vision.read_frames(p)
    events = vision.detect_motion(frames)
    assert len(events) >= 1
    assert events[0].peak_area > 0


def test_empty_clip_has_no_events_and_retries(tmp_path):
    p = make_clip(str(tmp_path / "x.avi"), "empty")
    run = TriageAgent(StubClassifier()).run(p)
    assert [e.action for e in run.events] == ["discard"]
    assert [s.tool for s in run.trace].count("detect_motion") == 2  # agent lowered the threshold once


def test_elephant_raises_alert_needing_approval(tmp_path):
    p = make_clip(str(tmp_path / "e.avi"), "elephant")
    run = TriageAgent(StubClassifier()).run(p)
    assert run.events[0].action == "alert_pending_approval"


def test_deer_is_logged(tmp_path):
    p = make_clip(str(tmp_path / "d.avi"), "deer")
    run = TriageAgent(StubClassifier()).run(p)
    assert run.events[0].action == "log"


def test_dark_blurry_clip_triggers_enhancement(tmp_path):
    p = make_clip(str(tmp_path / "n.avi"), "deer", dark=True, blur=3)
    run = TriageAgent(StubClassifier()).run(p)
    tools = [s.tool for s in run.trace]
    assert "enhance" in tools  # visual evidence changed the next tool call
    assert run.events, "event still found at night"


def test_api_demo_and_upload(tmp_path):
    c = TestClient(app)
    assert c.get("/healthz").json()["ok"]
    r = c.post("/v1/demo").json()
    assert r["events"] and r["trace"]
    p = make_clip(str(tmp_path / "u.avi"), "deer")
    with open(p, "rb") as fh:
        r = c.post("/v1/triage", files={"files": ("u.avi", fh, "video/x-msvideo")})
    assert r.status_code == 200 and r.json()["events"][0]["action"] == "log"
    r = c.post("/v1/triage", files={"files": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


class FakeDetector:
    name = "fake-detector"

    def __init__(self, dets):
        self.dets, self.calls = dets, 0

    def detect(self, img, min_score=0.2):
        self.calls += 1
        return self.dets


def _burst(tmp_path, moving):
    import numpy as np
    paths = []
    for i in range(3):
        img = np.full((360, 480, 3), 90, np.uint8)
        x = 60 + (120 * i if moving else 0)
        cv2.rectangle(img, (x, 150), (x + 80, 230), (40, 40, 200), -1)  # a red "elephant"
        f = tmp_path / f"b{i}.jpg"
        cv2.imwrite(str(f), img)
        paths.append(str(f))
    return paths


def test_burst_of_stills_finds_moving_animal(tmp_path):
    run = TriageAgent(StubClassifier(), detector=None).run(_burst(tmp_path, moving=True))
    assert any(s.tool == "detect_motion" and s.observation["events"] for s in run.trace)
    assert run.events[0].action == "alert_pending_approval"


def test_still_animal_falls_back_to_detector(tmp_path):
    from trailwatch.classify import Detection
    det = FakeDetector([Detection("animal", 0.9, (60, 150, 80, 80))])
    run = TriageAgent(StubClassifier(), detector=det).run(_burst(tmp_path, moving=False))
    tools = [s.tool for s in run.trace]
    assert "detect_animals" in tools and det.calls == 1  # no motion changed the next tool call
    assert run.events[0].action == "alert_pending_approval"
