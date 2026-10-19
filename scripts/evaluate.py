"""Synthetic benchmark: the agent loop against a fixed single-pass pipeline.

Clips vary by animal (elephant, deer, unknown, empty), light (day, night) and
blur. The single-pass baseline takes the peak frame once, with no enhancement
and no retries, which is what most fixed pipelines do.

Usage: python scripts/evaluate.py [n_per_cell]
"""
from __future__ import annotations

import itertools
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trailwatch import vision  # noqa: E402
from trailwatch.agent import TriageAgent  # noqa: E402
from trailwatch.classify import StubClassifier  # noqa: E402
from trailwatch.synth import make_clip  # noqa: E402

EXPECTED = {"elephant": "alert_pending_approval", "deer": "log", "unknown": "review", "empty": None}


def baseline(agent: TriageAgent, path: str):
    frames, _ = vision.read_frames(path)
    evs = vision.detect_motion(frames)
    if not evs:
        return None
    ev = evs[0]
    f = next(fr for fr in frames if fr.index == ev.peak_frame)
    pred = agent.clf.classify(vision.crop(f.image, ev.box))
    return agent._decide(pred, vision.assess_quality(f.image))[0]


def main(n: int = 5) -> dict:
    agent = TriageAgent(StubClassifier())
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for kind, dark, blur in itertools.product(EXPECTED, (False, True), (0, 3)):
            for s in range(n):
                p = make_clip(f"{d}/{kind}{dark}{blur}{s}.avi", kind, dark=dark, blur=blur, seed=s)
                t0 = time.perf_counter()
                run = agent.run(p)
                ms = (time.perf_counter() - t0) * 1000
                got = run.events[0].action if run.events else None
                tries = run.events[0].attempts if run.events else 0
                enh = sum(1 for st in run.trace if st.tool == "enhance")
                rows.append({"kind": kind, "dark": dark, "blur": blur, "expected": EXPECTED[kind],
                             "agent": got, "baseline": baseline(agent, p), "attempts": tries,
                             "enhanced": enh, "ms": round(ms, 1)})

    def acc(key, sub):
        return round(sum(r[key] == r["expected"] for r in sub) / len(sub), 3) if sub else None

    hard = [r for r in rows if r["dark"] or r["blur"]]

    def unsafe(key, sub):
        # Unsafe = a missed or wrong alert, or an automatic log of something that should not be logged.
        # Sending a clip to human review instead of logging it is slower but safe.
        bad = 0
        for r in sub:
            exp, got = r["expected"], r[key]
            if exp == got or (exp == "log" and got == "review"):
                continue
            bad += 1
        return round(bad / len(sub), 3) if sub else None
    summary = {
        "clips": len(rows),
        "agent_accuracy": acc("agent", rows),
        "baseline_accuracy": acc("baseline", rows),
        "agent_accuracy_dark_or_blurred": acc("agent", hard),
        "baseline_accuracy_dark_or_blurred": acc("baseline", hard),
        "agent_unsafe_error_rate": unsafe("agent", rows),
        "baseline_unsafe_error_rate": unsafe("baseline", rows),
        "agent_unsafe_dark_or_blurred": unsafe("agent", hard),
        "baseline_unsafe_dark_or_blurred": unsafe("baseline", hard),
        "mean_attempts": round(sum(r["attempts"] for r in rows) / len(rows), 2),
        "mean_ms_per_clip": round(sum(r["ms"] for r in rows) / len(rows), 1),
        "failures": [r for r in rows if r["agent"] != r["expected"]][:10],
    }
    return summary


if __name__ == "__main__":
    out = main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
    Path("docs").mkdir(exist_ok=True)
    Path("docs/eval_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "failures"}, indent=2))
    print("failures:", len(out["failures"]))
