"""HTTP endpoint: upload a clip, get the agent's decisions and full trace back."""
from __future__ import annotations

import os
import tempfile
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .agent import TriageAgent
from .synth import make_clip

app = FastAPI(title="TrailWatch", version="0.1.0")
agent = TriageAgent()
MAX_BYTES = int(os.environ.get("TRAILWATCH_MAX_BYTES", 50 * 1024 * 1024))

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>TrailWatch</title><style>body{font:15px system-ui;max-width:760px;margin:2em auto;padding:0 16px}
pre{background:#f4f4f4;padding:12px;overflow:auto;max-height:60vh}</style>
<h1>TrailWatch</h1><p>Upload a short camera-trap clip (AVI or MP4, up to 50 MB). The agent finds motion with OpenCV 5,
checks image quality, retries with enhancement or another frame when unsure, identifies the animal and decides: log it,
send it for human review, or raise an alert that waits for approval.</p>
<form id=f><input type=file name=file accept="video/*"> <button>Analyse</button>
<button type=button id=demo>Run synthetic demo</button></form><pre id=o></pre>
<script>
const o=document.getElementById('o');
async function show(r){o.textContent=JSON.stringify(await r.json(),null,2)}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();o.textContent='Analysing...';
 show(await fetch('/v1/triage',{method:'POST',body:new FormData(e.target)}))};
document.getElementById('demo').onclick=async()=>{o.textContent='Analysing...';show(await fetch('/v1/demo',{method:'POST'}))};
</script>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.get("/healthz")
def health() -> dict:
    return {"ok": True, "classifier": agent.clf.name}


def _summary(run) -> dict:
    d = run.to_dict()
    return {"classifier": d["classifier"], "frames": d["frames_read"],
            "events": [{k: e[k] for k in ("action", "reason", "prediction", "attempts", "frame_used", "quality")}
                       for e in d["events"]],
            "trace": d["trace"]}


@app.post("/v1/triage")
async def triage(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "clip too large")
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".mp4"
    if suffix not in (".mp4", ".avi", ".mov", ".mkv"):
        raise HTTPException(415, "unsupported video type")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, f"{uuid.uuid4().hex}{suffix}")
        with open(p, "wb") as fh:
            fh.write(data)
        try:
            run = agent.run(p)
        except ValueError as e:
            raise HTTPException(422, str(e))
    return _summary(run)


@app.post("/v1/demo")
def demo() -> dict:
    with tempfile.TemporaryDirectory() as d:
        p = make_clip(os.path.join(d, "demo.avi"), "elephant", dark=True, blur=2)
        return _summary(agent.run(p))
