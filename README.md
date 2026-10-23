# TrailWatch

**An OpenCV 5 agent that triages camera-trap footage: it finds the animal, checks whether the picture is good enough, asks a second model when its tools disagree, and decides what should happen next.**

Built for the OpenCV AI Competition 2026 (powered by AWS), Agentic Vision track. Open source under Apache-2.0.

Camera traps record thousands of triggers a week; most are wind or empty frames, and a few matter, such as an elephant walking towards a village at night. Fixed "AI says X" pipelines fail quietly on animals that stand still, dark infrared frames and look-alike species. TrailWatch treats each trigger as a small investigation: the measurements OpenCV 5 makes decide the next tool call, and every alert waits for a person.

## Results on real camera-trap data

260 labelled bursts from Snapshot Serengeti (LILA BC), 12 species plus empty, same models in both columns ([full report](docs/REPORT.md)):

| | Fixed single pass | TrailWatch agent |
| --- | --- | --- |
| Correct action | 71.9% | **92.7%** |
| Animals wrongly discarded | 68 of 240 | **8 of 240** |
| Elephants, lions, leopards alerted | 37 of 60 | **50 of 60** |
| Watch-listed animals reaching a person | 40 of 60 | **57 of 60** |
| False alerts | 0 | 0 |

## How it works

1. **Find motion** with OpenCV 5: MOG2 for video, a median-background difference for bursts of stills. Nothing? Retry with a lower threshold, then ask **MegaDetector** (it finds animals that stood still).
2. **Check quality**: brightness, Laplacian sharpness, infrared test. Poor? **Enhance** (adaptive gamma, CLAHE, unsharp mask), re-measure, keep whichever version the classifier is surer about.
3. **Identify** the crop with Google's **SpeciesNet**. Unsure? Re-classify the enhanced crop, then the next sharpest frame. "Blank" although something moved? Ask MegaDetector and re-crop.
4. **Decide**: `log`, `review` (unsure, or a look-alike of a watch-listed species), `alert_pending_approval` (watch-listed species, held for a person), or `discard` (empty, people, vehicles: not kept).

Every step is returned as a trace of `{tool, inputs, observation, decision}`. See [docs/architecture.md](docs/architecture.md) and example traces in [docs/sample_traces.json](docs/sample_traces.json).

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                   # 9 tests, no model download
uvicorn trailwatch.api:app --port 8080      # stub classifier; open http://localhost:8080
```

With the real models (about 1 GB of PyTorch and weights):

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-speciesnet.txt
TRAILWATCH_CLASSIFIER=speciesnet uvicorn trailwatch.api:app --port 8080
```

Or with Docker (models baked in): `docker build -t trailwatch . && docker run -p 8080:8080 trailwatch`

### API

- `GET /` demo page with six real sample bursts
- `GET /healthz`
- `POST /v1/triage` with `files`: one video (`.mp4`, `.avi`, `.mov`, `.mkv`) or 2 to 10 stills (`.jpg`, `.png`) from one trigger, up to 50 MB
- `GET /v1/samples`, `POST /v1/samples/{id}`: run a bundled sample burst
- Watch list: `TRAILWATCH_WATCHLIST="Elephas maximus=elephant near the village;Panthera tigris=tiger"`

## Deploy on AWS (Graviton)

One Amazon Linux 2023 `t4g.medium` instance (AWS Graviton, arm64) started with [deploy/ec2-user-data.sh](deploy/ec2-user-data.sh): it installs Docker, builds the image and serves it on port 80. Uploads are capped at 50 MB, processed one at a time and not kept.

## Responsible use

- Alerts never leave the system without a person approving them; unsure and look-alike cases go to review.
- People and vehicles are discarded, never stored.
- Classifier scores are not calibrated probabilities; tune thresholds and the watch list per site from a reviewed sample.
- Generalise the locations of sensitive species before publishing any data.

## Credits

Sample images: Snapshot Serengeti (Swanson et al. 2015) via LILA BC, CDLA-Permissive 1.0. Models: SpeciesNet (Google, Apache-2.0), MegaDetector v5a (MIT). Part of [WildNetwork](https://wildnetwork.arunrajiah.com), an open live map of wildlife detections.
