# TrailWatch: technical report

**An OpenCV 5 agent that triages camera-trap footage and decides what should happen next.**
OpenCV AI Competition 2026, Agentic Vision track. Author: Arun Rajiah (Wildlife Network, Chennai). Code: Apache-2.0.

## 1. Problem and users

Camera traps at forest edges, farms and wetlands record thousands of triggers a week. Most are wind, rain or empty frames; a few
matter, such as an elephant walking towards a village at night or a leopard near livestock. Small reserves, community
conservation groups and researchers cannot look at every frame, and fixed "AI says X" pipelines fail quietly on the hard cases:
animals that stand still, dark infrared frames, and look-alike species.

**Users:** reserve staff and rangers who need alerts they can trust; ecologists who need clean, labelled records; and
community monitoring networks such as [Wildlife Network](https://wlnw.org), an open live map of wildlife
detections that is installing its first stations in Chennai, India, in December 2026.

## 2. What the agent does

TrailWatch treats each trigger as a small investigation. Its tools are OpenCV 5 functions plus two models, and the
measurements from one step choose the next step. The full loop is in [architecture.md](architecture.md).

1. **Find motion.** Video: MOG2 background subtraction, morphology and contours. Bursts of stills (2 to 10 frames, too few for
   MOG2): a per-pixel median of the burst as the background, the largest colour-channel difference per frame, with the median
   difference subtracted so exposure changes are not mistaken for animals. No motion: retry once with a lower threshold.
2. **Still nothing:** call **MegaDetector v5a**, which finds animals that did not move. This is slower, so it is only called
   when the cheap tool fails.
3. **Check quality** of the chosen frame: mean brightness, variance of the Laplacian (sharpness) and an infrared test (near-zero
   saturation).
4. **Recover poor frames:** adaptive gamma towards a target brightness, CLAHE on the L channel, and an unsharp mask. Re-measure;
   classify both versions and keep the one the classifier is surer about (enhancement can distort colour).
5. **Identify** the crop with **SpeciesNet v4.0.3a** (Google's open camera-trap classifier, about 2,500 labels).
6. **Resolve disagreements.** Motion found but the classifier says "blank" or is unsure: ask MegaDetector for the animal's own
   box (motion boxes can cover only the part that moved), re-crop and keep the better answer, or discard if it agrees nothing
   is there. Unsure: re-classify the enhanced crop, then the next sharpest frame (up to 3 tries).
7. **Decide an action**:
   - `log`: confident identification;
   - `review`: unsure, or the top answer is a look-alike of a watch-listed species (for example cheetah with leopard close
     behind);
   - `alert_pending_approval`: a watch-listed species, held until a person approves it;
   - `discard`: empty frames, people and vehicles (not kept, for privacy).

Every step is written to a trace (`tool`, `inputs`, `observation`, `decision`) that the API returns with the result.

## 3. OpenCV 5 implementation

`opencv-python-headless==5.0.0.93`. Functions used: `VideoCapture`, `createBackgroundSubtractorMOG2`, `GaussianBlur`,
`absdiff`, `threshold`, `morphologyEx`, `dilate`, `findContours`, `boundingRect`, `contourArea`, `cvtColor` (BGR, grey, HSV,
LAB), `Laplacian`, `LUT` (adaptive gamma), `createCLAHE`, `addWeighted` (unsharp mask), `resize`, `imread`, `imwrite`.
All perception code is in [`trailwatch/vision.py`](../trailwatch/vision.py); the agent loop is in
[`trailwatch/agent.py`](../trailwatch/agent.py).

## 4. AWS deployment

One EC2 instance on **AWS Graviton** (`t4g.medium`, arm64, Amazon Linux 2023, 30 GB gp3), started with
[`deploy/ec2-user-data.sh`](../deploy/ec2-user-data.sh). It installs Docker, builds the multi-arch image (Python 3.12,
OpenCV 5, PyTorch CPU, SpeciesNet and MegaDetector weights baked in at build time) and serves the FastAPI app on port 80.

- The endpoint accepts one video or 2 to 10 stills, caps uploads at 50 MB, processes one request at a time and keeps nothing
  after the request ends.
- The models run on the Arm CPU; no GPU is needed. A burst takes about 2 seconds of model time on a 2-vCPU machine.
- The security group opens only port 80 (HTTP).

## 5. Evaluation

### Real camera-trap data

**260 labelled bursts** (3 stills each) from **Snapshot Serengeti** (LILA BC), 20 per class across 13 classes (12 species
plus empty), sampled at random from seasons 1 to 11. Labels are the volunteer consensus. Human images are withheld from the
public dataset for privacy, so people are not part of this test.

Watch list for this deployment: African elephant, lion, leopard. The **baseline** uses the same models and decision rules in a
fixed single pass (motion on the burst, one crop of the peak frame, one classification), with no enhancement, retries or detector
checks. Script: [`scripts/evaluate_real.py`](../scripts/evaluate_real.py); sample list: [`eval_real_sample.json`](eval_real_sample.json); results: [`eval_real.json`](eval_real.json).

| Metric | Baseline | TrailWatch agent |
| --- | --- | --- |
| Correct action | 71.9% | **93.5%** |
| Animals wrongly discarded as empty | 68 of 240 | **8 of 240** |
| Watch-listed animals alerted | 37 of 60 | **52 of 60** |
| Watch-listed animals reaching a person (alert or review) | 40 of 60 | **56 of 60** |
| False alerts | 0 | 0 |
| Auto-logged records with the right genus | 119 of 122 | 159 of 165 |
| Sent to human review | 14 | 16 |

Correct action by class (agent / baseline, out of 20): baboon 19/11, buffalo 20/16, cheetah 19/18, elephant 20/17, empty 19/19,
Thomson's gazelle 19/12, giraffe 18/17, spotted hyena 18/14, impala 19/14, leopard 13/7, lion 19/13, wildebeest 20/17,
zebra 20/12.

Tool use across the 260 bursts: 331 motion detections, 102 MegaDetector calls, 25 re-classifications of enhanced crops,
8 enhancements. Mean time per burst about 2.5 seconds on 2 CPU cores, including both models.

**What the agent changes.** Most of the gain comes from two decisions: calling the detector when motion detection finds
nothing (animals that stood still across the burst), and checking "blank" answers with the detector when motion was seen.
Together they cut animals wrongly discarded from 68 to 8. A third use of the detector, asking it for the animal's own box when the classifier is unsure about a loose motion box, raised alerts caught from 50 to 52 of 60. The look-alike rule sends spotted cats with a leopard close behind to
a person instead of logging them.

### Synthetic clips

[`scripts/evaluate.py`](../scripts/evaluate.py) runs 80 synthetic video clips (elephant, deer, unknown, empty; day and night;
sharp and blurred) with a deterministic stub classifier, to test the video path and the retry logic. Agent and baseline both
reach 81.2% there, with no unsafe errors: the synthetic data is too easy to separate them, which is why the real-data test above
matters more.

### Failure cases

- **Leopard read as cheetah** (7 of 20 leopards): a classifier error on similar spotted cats. The look-alike rule sent 4 of
  them to review; 3 were still logged as cheetahs with scores of 0.76 to 0.92. One more leopard was alerted as a lion.
- **Animals still missed** (8 of 240): small, distant or hidden animals that neither motion detection nor MegaDetector found at
  the 0.2 score threshold (baboon, cheetah, gazelle, two giraffes, two hyenas, impala).
- **Wrong species, confident** (6 of 165 auto-logged): a buffalo as wildebeest, a wildebeest as cattle, a lion as hartebeest,
  and the three leopards above.
- **One empty frame** sent to review as impala (score 0.52).

## 6. Limitations

- Tested on one African savanna dataset. The first deployment is in India, where SpeciesNet has less training data for local
  species; the watch list and thresholds must be tuned per site from a reviewed sample.
- Classifier scores are not calibrated probabilities; `accept_score` and `review_score` are policy settings.
- SpeciesNet's geofencing step is not used, so out-of-range species (for example cattle in a reserve) can appear.
- One request at a time on one small instance: fine for a field station uploading triggers, not for bulk archives.

## 7. Responsible use

- **Human control:** alerts never leave the system without a person approving them, and unsure or look-alike cases go to review.
- **Privacy:** people and vehicles are discarded, never stored or classified further. The endpoint keeps no uploads.
- **Sensitive species:** locations of watch-listed species should be generalised before any data is published (Wildlife Network
  rounds coordinates on the device).
- **Honest reporting:** the trace shows every step, so a reviewer can see why a decision was made and challenge it.

## 8. Reproduce

```bash
pip install -r requirements-dev.txt && pytest -q          # 9 tests
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-speciesnet.txt
TRAILWATCH_CLASSIFIER=speciesnet python scripts/evaluate_real.py docs/eval_real_sample.json /path/to/snapshot-serengeti
```

The sample list ([`eval_real_sample.json`](eval_real_sample.json)) holds the 260 sequence ids and file names; images are public at LILA BC.

## Data and model credits

- Snapshot Serengeti: Swanson et al. (2015), *Scientific Data* 2: 150026, via [LILA BC](https://lila.science/datasets/snapshot-serengeti), CDLA-Permissive 1.0.
- SpeciesNet: Google, Apache-2.0. MegaDetector v5a: Microsoft AI for Earth and contributors, MIT.
