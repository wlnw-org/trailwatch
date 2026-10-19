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
    assert run.events == []
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
        r = c.post("/v1/triage", files={"file": ("u.avi", fh, "video/x-msvideo")})
    assert r.status_code == 200 and r.json()["events"][0]["action"] == "log"
    r = c.post("/v1/triage", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415
