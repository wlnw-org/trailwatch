"""HTTP endpoint: upload a clip or a burst of stills, get the agent's decisions and full trace back."""
from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
import cv2
from fastapi.responses import FileResponse, HTMLResponse

from . import vision
from .agent import TriageAgent, policy_from_env
from .synth import make_clip

app = FastAPI(title="TrailWatch", version="0.2.0")
agent = TriageAgent(policy=policy_from_env())
_lock = threading.Lock()  # one analysis at a time keeps memory flat on a small instance
MAX_BYTES = int(os.environ.get("TRAILWATCH_MAX_BYTES", 50 * 1024 * 1024))
VIDEO = (".mp4", ".avi", ".mov", ".mkv")
IMAGES = (".jpg", ".jpeg", ".png")
SAMPLES = Path(os.environ.get("TRAILWATCH_SAMPLES", Path(__file__).resolve().parents[1] / "samples"))
SERENGETI = {"Loxodonta africana": "elephant near the station", "Panthera leo": "lion recorded",
             "Panthera pardus": "leopard recorded"}

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>TrailWatch</title><style>
body{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:2em auto;padding:0 16px;color:#1d2a22;background:#f7f5ee}
h1{margin:.2em 0}.muted{color:#5b665f}button{font:inherit;padding:.4em .8em;margin:.2em;border:1px solid #8a9a8f;border-radius:6px;background:#fff;cursor:pointer}
.card{background:#fff;border:1px solid #ddd8cc;border-radius:10px;padding:14px 16px;margin:12px 0}
.act{font-weight:700;padding:2px 8px;border-radius:6px;background:#e9efe7}.alert{background:#f6d9bc}
ol{padding-left:1.3em}li{margin:.25em 0}code{background:#eee;padding:0 4px;border-radius:4px}img{max-width:100%;border-radius:6px}
</style>
<h1>TrailWatch</h1>
<p class=muted>An OpenCV 5 agent that triages camera-trap footage. It finds motion, checks image quality, enhances or retries
when the picture is poor, asks an animal detector when its tools disagree, identifies the species, and decides: log it, send it
to a person, discard it (people, empty frames), or raise an alert that waits for a person to approve it.</p>
<div class=card><b>Try real camera-trap bursts</b> <span class=muted>(Snapshot Serengeti, LILA BC)</span><div id=s></div></div>
<div class=card><b>Upload your own</b>: a clip (MP4, AVI, MOV, MKV) or 2 to 10 stills from one trigger (JPG, PNG), up to 50 MB.
<form id=f><input type=file name=files multiple accept="video/*,image/*"> <button>Analyse</button></form></div>
<div id=o></div>
<script>
const o=document.getElementById('o');
function esc(t){return String(t).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function show(r){if(!r.ok){o.innerHTML='<div class=card>Error '+r.status+': '+esc(await r.text())+'</div>';return}
 const j=await r.json();let h='';
 for(const e of j.events){const p=e.prediction||{};h+='<div class=card><span class="act '+(e.action.startsWith('alert')?'alert':'')+'">'+esc(e.action)+
  '</span> '+esc(e.reason)+(p.label?'<br>'+esc(p.label)+(p.scientific?' (<i>'+esc(p.scientific)+'</i>)':'')+', score '+p.score:'')+
  (e.annotated&&e.annotated.startsWith('data:image/jpeg;base64,')?'<br><img src="'+e.annotated+'" alt="frame used, with the box the agent classified">':'')+'</div>'}
 h+='<div class=card><b>Agent trace</b> <span class=muted>('+j.trace.length+' steps, '+esc(j.classifier)+')</span><ol>';
 for(const s of j.trace){h+='<li><code>'+esc(s.tool)+'</code> '+esc(s.decision)+'</li>'}
 o.innerHTML=h+'</ol></div>'}
fetch('/v1/samples').then(r=>r.json()).then(l=>{document.getElementById('s').innerHTML=l.map(x=>
 '<button data-id="'+x.id+'">'+esc(x.title)+'</button>').join('');
 document.querySelectorAll('#s button').forEach(b=>b.onclick=async()=>{o.innerHTML='<div class=card>Analysing (a few seconds)...</div>';
 const id=b.dataset.id,x=l.find(y=>y.id==id);
 show(await fetch('/v1/samples/'+id,{method:'POST'}));
 o.insertAdjacentHTML('beforeend','<div class=card>'+x.files.map(f=>'<img src="/samples/'+id+'/'+f+'" alt="frame">').join('')+'</div>')})});
document.getElementById('f').onsubmit=async e=>{e.preventDefault();o.innerHTML='<div class=card>Analysing...</div>';
 show(await fetch('/v1/triage',{method:'POST',body:new FormData(e.target)}))};
</script>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.get("/healthz")
def health() -> dict:
    return {"ok": True, "classifier": agent.clf.name, "detector": agent.det.name if agent.det else None}


def _annotate(stills: list[str] | None, e: dict) -> str | None:
    """The frame the agent used, with its box drawn, as a small JPEG data URL (stills only)."""
    if not stills or e.get("frame_used") is None or not e.get("box"):
        return None
    frames = vision.read_images([stills[e["frame_used"]]])
    if not frames:
        return None
    img = frames[0].image.copy()
    x, y, w, h = e["box"]
    colour = (32, 96, 200) if e["action"].startswith("alert") else (60, 140, 60)
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, 3)
    label = ((e.get("prediction") or {}).get("label") or "")[:40]
    cv2.putText(img, f"{e['action']}: {label}", (x, max(28, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def _summary(run, stills: list[str] | None = None) -> dict:
    d = run.to_dict()
    events = []
    for e in d["events"]:
        out = {k: e[k] for k in ("action", "reason", "prediction", "attempts", "frame_used", "quality", "box")}
        out["annotated"] = _annotate(stills, e)
        events.append(out)
    return {"classifier": d["classifier"], "frames": d["frames_read"], "events": events, "trace": d["trace"]}


def _run(source, policy: dict | None = None):
    with _lock:
        saved = agent.policy
        if policy:
            agent.policy = {**saved, **policy}
        try:
            return agent.run(source)
        finally:
            agent.policy = saved


@app.post("/v1/triage")
async def triage(files: list[UploadFile] = File(...)) -> dict:
    if not files or len(files) > 10:
        raise HTTPException(400, "send one video or 2 to 10 images")
    with tempfile.TemporaryDirectory() as d:
        paths, total = [], 0
        for f in files:
            data = await f.read()
            total += len(data)
            if total > MAX_BYTES:
                raise HTTPException(413, "upload too large")
            suffix = os.path.splitext(f.filename or "")[1].lower()
            if suffix not in VIDEO + IMAGES:
                raise HTTPException(415, "unsupported file type")
            p = os.path.join(d, f"{len(paths):02d}_{uuid.uuid4().hex}{suffix}")
            with open(p, "wb") as fh:
                fh.write(data)
            paths.append(p)
        kinds = {os.path.splitext(p)[1] in VIDEO for p in paths}
        if len(kinds) > 1 or (True in kinds and len(paths) > 1):
            raise HTTPException(400, "send one video, or images only")
        try:
            stills = None if True in kinds else paths
            run = _run(paths[0] if True in kinds else paths)
            return _summary(run, stills)
        except ValueError as e:
            raise HTTPException(422, str(e))


def _samples() -> list[dict]:
    f = SAMPLES / "samples.json"
    return json.loads(f.read_text()) if f.exists() else []


@app.get("/v1/samples")
def samples() -> list[dict]:
    return _samples()


@app.post("/v1/samples/{sid}")
def run_sample(sid: str) -> dict:
    s = next((x for x in _samples() if x["id"] == sid), None)
    if s is None:
        raise HTTPException(404, "no such sample")
    paths = [str(SAMPLES / sid / f) for f in s["files"]]
    return _summary(_run(paths, {"watchlist": SERENGETI}), paths)


@app.get("/samples/{sid}/{name}")
def sample_file(sid: str, name: str):
    s = next((x for x in _samples() if x["id"] == sid), None)
    if s is None or name not in s["files"]:
        raise HTTPException(404, "not found")
    return FileResponse(SAMPLES / sid / name, media_type="image/jpeg")


@app.post("/v1/demo")
def demo() -> dict:
    with tempfile.TemporaryDirectory() as d:
        p = make_clip(os.path.join(d, "demo.avi"), "elephant", dark=True, blur=2)
        return _summary(_run(p))
