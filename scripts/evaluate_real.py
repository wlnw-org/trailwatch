"""Real-data benchmark on labelled camera-trap bursts (Snapshot Serengeti, LILA BC).

Each item is one trigger burst of three stills with a volunteer consensus label.
We compare the agent against a fixed single-pass pipeline that uses the same
models: motion on the burst, one crop of the peak frame, one classification,
the same decision rules, and no enhancement, retries or detector checks.

Usage:
  TRAILWATCH_CLASSIFIER=speciesnet python scripts/evaluate_real.py SAMPLE.json IMAGE_ROOT [OUT.json]
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trailwatch import vision  # noqa: E402
from trailwatch.agent import TriageAgent  # noqa: E402

WATCHLIST = {  # the alert policy for a Serengeti deployment
    "Loxodonta africana": "elephant near the station",
    "Panthera leo": "lion recorded",
    "Panthera pardus": "leopard recorded",
}
GENUS = {"elephant": "loxodonta", "lion": "panthera", "leopard": "panthera", "cheetah": "acinonyx",
         "zebra": "equus", "wildebeest": "connochaetes", "gazellethomsons": "eudorcas", "impala": "aepyceros",
         "buffalo": "syncerus", "giraffe": "giraffa", "hyenaspotted": "crocuta", "baboon": "papio"}
ALERT = {"elephant", "lion", "leopard"}


def expected(label: str) -> str:
    if label in ("empty", "human"):
        return "discard"
    return "alert_pending_approval" if label in ALERT else "log_or_review"


def baseline(agent: TriageAgent, paths: list[str]) -> dict:
    frames = vision.read_images(paths)
    evs = vision.detect_motion_burst(frames)
    if not evs:
        return {"action": "discard", "label": "", "scientific": "", "score": 0.0}
    ev = evs[0]
    f = next(fr for fr in frames if fr.index == ev.peak_frame)
    pred = agent.clf.classify(vision.crop(f.image, ev.box))
    action, _ = agent._decide(pred, vision.assess_quality(f.image))
    return {"action": action, "label": pred.label, "scientific": pred.scientific, "score": pred.score}


def grade(label: str, got: dict) -> dict:
    exp, act = expected(label), got["action"]
    sci = (got.get("scientific") or "").lower()
    ok = act == exp or (exp == "log_or_review" and act in ("log", "review"))
    return {
        "correct_action": ok,
        "missed_animal": label not in ("empty", "human") and act == "discard",
        "missed_alert": label in ALERT and act != "alert_pending_approval",
        "watch_seen": label in ALERT and act in ("alert_pending_approval", "review"),
        "false_alert": label not in ALERT and act.startswith("alert"),
        "person_kept": label == "human" and act != "discard",
        "logged": act == "log",
        "logged_right": act == "log" and label in GENUS and sci.startswith(GENUS[label]),
        "review": act == "review",
    }


def summarise(rows: list[dict], key: str) -> dict:
    n = len(rows)
    g = [r[key + "_grade"] for r in rows]
    s = lambda k: sum(x[k] for x in g)  # noqa: E731
    animals = [r for r in rows if r["label"] not in ("empty", "human")]
    alerts = [r for r in rows if r["label"] in ALERT]
    humans = [r for r in rows if r["label"] == "human"]
    logged = s("logged")
    return {
        "bursts": n,
        "correct_action": round(s("correct_action") / n, 3),
        "missed_animals": f"{s('missed_animal')} of {len(animals)}",
        "watchlist_alerts_caught": f"{len(alerts) - s('missed_alert')} of {len(alerts)}",
        "watchlist_reaching_a_person": f"{s('watch_seen')} of {len(alerts)}",
        "false_alerts": s("false_alert"),
        "people_kept": f"{s('person_kept')} of {len(humans)}",
        "auto_logged": logged,
        "auto_logged_correct_genus": f"{s('logged_right')} of {logged}",
        "sent_to_review": s("review"),
    }


def main(sample: str, root: str, out: str = "docs/eval_real.json") -> dict:
    items = json.load(open(sample))
    agent = TriageAgent(policy={"watchlist": WATCHLIST})
    rows = []
    t0 = time.time()
    for i, it in enumerate(items):
        paths = [str(Path(root) / f) for f in it["files"]]
        ta = time.perf_counter()
        run = agent.run(paths)
        ms = (time.perf_counter() - ta) * 1000
        top = max(run.events, key=lambda e: (e.action.startswith("alert"), (e.prediction or {}).get("score", 0)))
        pa = top.prediction or {}
        a = {"action": top.action, "label": pa.get("label", ""), "scientific": pa.get("scientific", ""),
             "score": pa.get("score", 0.0)}
        b = baseline(agent, paths)
        tools = collections.Counter(s.tool for s in run.trace)
        rows.append({"seq": it["seq"], "label": it["label"], "agent": a, "baseline": b,
                     "agent_grade": grade(it["label"], a), "baseline_grade": grade(it["label"], b),
                     "tools": dict(tools), "ms": round(ms)})
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(items)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    by_label = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        by_label[r["label"]]["agent_ok"] += r["agent_grade"]["correct_action"]
        by_label[r["label"]]["baseline_ok"] += r["baseline_grade"]["correct_action"]
        by_label[r["label"]]["n"] += 1
    result = {
        "dataset": "Snapshot Serengeti v2.1 (LILA BC), seasons 1 to 11, 3-frame bursts, consensus labels",
        "models": {"classifier": agent.clf.name, "detector": agent.det.name if agent.det else None},
        "agent": summarise(rows, "agent"),
        "baseline": summarise(rows, "baseline"),
        "tool_use": dict(sum((collections.Counter(r["tools"]) for r in rows), collections.Counter())),
        "mean_ms_per_burst_agent": round(sum(r["ms"] for r in rows) / len(rows)),
        "by_label": {k: dict(v) for k, v in sorted(by_label.items())},
        "rows": rows,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return result


if __name__ == "__main__":
    main(*sys.argv[1:])
