# TrailWatch

**An OpenCV 5 agent that triages camera-trap video: it finds the animal, checks whether the picture is good enough, tries again when it is not, and decides what should happen next.**

Built for the OpenCV AI Competition 2026 (powered by AWS). Open source under Apache-2.0.

Camera traps at forest edges, farms and wetlands record thousands of clips a week. Most are wind, rain or empty frames; a few are animals that matter, such as an elephant walking towards a village at night. Volunteers and small reserves cannot watch everything, and fixed "AI says X" pipelines quietly fail on dark, blurry infrared footage. TrailWatch treats each clip as a small investigation: the measurements OpenCV 5 makes decide the next tool call.

## What it does

1. **Find motion** with OpenCV 5 background subtraction (MOG2), morphology and contours. If nothing is found, the agent lowers the threshold once and looks again.
2. **Check quality** of the best frame: brightness, sharpness (variance of the Laplacian) and an infrared night test (colour saturation).
3. **Recover poor frames**: CLAHE contrast and an unsharp mask, then re-measure. If the frame is still poor, the agent moves to the next sharpest frame of the same event.
4. **Identify** the cropped animal with a pluggable classifier: Google's open **SpeciesNet** model, or a fast deterministic stub for tests and the demo.
5. **Decide an action**:
   - `log`: confident identification, kept as a record;
   - `review`: unsure, sent to a person (for example in SpeciesNet Studio);
   - `alert_pending_approval`: a watch-listed species (elephant, tiger, leopard, mugger crocodile), held until a person approves the alert;
   - `discard`: people, vehicles or empty frames, not kept, for privacy.

Every step is written to a trace (`tool`, `inputs`, `observation`, `decision`), so a reviewer can see exactly why the agent did what it did.

```
read_frames -> detect_motion --(no events)--> detect_motion(sensitive)
                   |
             for each event:
             best_frame -> assess_quality --(poor)--> enhance -> assess_quality --(still poor)--> next frame
                   |
               classify --(score < accept)--> next frame / enhanced crop (up to 3 tries)
                   |
               decide -> log | review | alert_pending_approval | discard
```

See [docs/architecture.md](docs/architecture.md) and the full [technical report](docs/REPORT.md).

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                   # 7 tests, synthetic clips only
uvicorn trailwatch.api:app --port 8080      # open http://localhost:8080
```

Use the real classifier (large download):

```bash
pip install -r requirements-speciesnet.txt
TRAILWATCH_CLASSIFIER=speciesnet uvicorn trailwatch.api:app --port 8080
```

API:

- `GET /healthz`
- `POST /v1/triage` with a video file (`.mp4`, `.avi`, `.mov`, `.mkv`, up to 50 MB)
- `POST /v1/demo` runs a synthetic night-time clip and returns the trace

## Deploy on AWS (Graviton)

The Docker image is multi-arch. The reference deployment is one Amazon Linux 2023 `t4g.small` instance (AWS Graviton, arm64) started with [deploy/ec2-user-data.sh](deploy/ec2-user-data.sh), which installs Docker, builds the image and serves it on port 80. The API only accepts video files, caps uploads at 50 MB and keeps nothing after a request finishes.

## Responsible use

- Alerts never leave the system without a person approving them.
- People and vehicles are discarded, never stored.
- Classifier scores are not calibrated probabilities; thresholds are policy settings and should be tuned per site from a reviewed sample.
- Locations of sensitive species should be generalised before any data is published.
