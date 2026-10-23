"""Species classifiers behind one small interface.

SpeciesNet (Google's open camera-trap model) is used when installed; a
deterministic stub keeps tests and the demo fast and offline.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Prediction:
    label: str          # common name, or "blank" / "human" / "vehicle"
    scientific: str
    score: float        # classifier score in [0, 1]; not a calibrated probability
    model: str
    alternatives: list = field(default_factory=list)  # next-best [scientific, score] pairs


class Classifier:
    name = "base"

    def classify(self, crop: np.ndarray) -> Prediction:  # pragma: no cover
        raise NotImplementedError


class StubClassifier(Classifier):
    """Colour-coded stand-in used by tests and the synthetic demo video.

    Red blobs are elephants, blue blobs are spotted deer, everything else is
    unknown. The score falls with blur and darkness, as a real model's does,
    which lets the agent's retry behaviour be exercised end to end.
    """

    name = "stub-colour-v1"

    def classify(self, crop: np.ndarray) -> Prediction:
        if crop.size == 0:
            return Prediction("blank", "", 0.0, self.name)
        b, g, r = (float(x) for x in crop.reshape(-1, 3).mean(axis=0))
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = min(1.0, float(cv2.Laplacian(grey, cv2.CV_64F).var()) / 400.0)
        bright = min(1.0, float(grey.mean()) / 90.0)
        q = 0.35 + 0.65 * sharp * bright
        tot = max(1.0, r + g + b)
        sr, sg, sb = r / tot, g / tot, b / tot  # chromaticity survives darkness, unlike raw differences
        if sr > max(sg, sb) + 0.05:
            return Prediction("Asian elephant", "Elephas maximus", round(0.95 * q, 3), self.name)
        if sb > max(sr, sg) + 0.05:
            return Prediction("Spotted deer", "Axis axis", round(0.92 * q, 3), self.name)
        return Prediction("unknown animal", "", round(0.3 * q, 3), self.name)


class SpeciesNetClassifier(Classifier):
    """Google SpeciesNet's species classifier, run on the crop the agent chose.

    The crop already comes from OpenCV motion detection, so the classifier sees
    it as one full-frame box. Geofencing (SpeciesNet's ensemble step) is not
    applied here; the agent's policy and human review play that role.
    """

    name = "speciesnet"

    def __init__(self, model: str | None = None):
        import speciesnet
        from speciesnet import SpeciesNetClassifier as _Clf

        self._bbox = speciesnet.BBox(0.0, 0.0, 1.0, 1.0)
        self.model_name = model or os.environ.get("SPECIESNET_MODEL", speciesnet.DEFAULT_MODEL)
        self.model = _Clf(self.model_name)
        ver = self.model_name.split("/")[-2] if "/" in self.model_name else self.model_name
        self.name = f"speciesnet:{ver}"

    def classify(self, crop: np.ndarray) -> Prediction:
        from PIL import Image

        if crop.size == 0:
            return Prediction("blank", "", 0.0, self.name)
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pre = self.model.preprocess(img, bboxes=[self._bbox])
        out = self.model.predict("crop", pre)
        cls = (out.get("classifications_list") or [{}])[0]
        classes, scores = cls.get("classes") or [""], cls.get("scores") or [0.0]
        # Labels look like "uuid;class;order;family;genus;species;common name"
        def parse(label: str) -> tuple[str, str]:
            parts = label.split(";")
            common = parts[-1] if parts and parts[-1] else "unknown"
            sci = " ".join(x for x in parts[4:6] if x).strip().capitalize() if len(parts) >= 6 else ""
            return common, ("" if common in ("blank", "human", "vehicle") else sci)

        common, sci = parse(classes[0])
        alts = [[parse(c)[1], round(float(sc), 4)] for c, sc in list(zip(classes, scores))[1:5] if parse(c)[1]]
        return Prediction(common, sci, round(float(scores[0]), 4), self.name, alts)


@dataclass
class Detection:
    label: str          # "animal", "human" or "vehicle"
    score: float
    box: tuple[int, int, int, int]  # x, y, w, h in pixels


class MegaDetector:
    """MegaDetector v5a (via the speciesnet package): finds animals, people and vehicles.

    Slower than OpenCV motion detection, so the agent calls it only when the
    cheap tools disagree or find nothing.
    """

    name = "megadetector-v5a"

    def __init__(self, model: str | None = None):
        import speciesnet
        from speciesnet import SpeciesNetDetector

        self.model = SpeciesNetDetector(model or os.environ.get("SPECIESNET_MODEL", speciesnet.DEFAULT_MODEL))

    def detect(self, img: np.ndarray, min_score: float = 0.2) -> list[Detection]:
        from PIL import Image

        h, w = img.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        out = self.model.predict("frame", self.model.preprocess(pil))
        dets = []
        for d in out.get("detections") or []:
            if d["conf"] < min_score:
                continue
            x, y, bw, bh = d["bbox"]
            dets.append(Detection(d["label"], round(float(d["conf"]), 4),
                                  (int(x * w), int(y * h), max(1, int(bw * w)), max(1, int(bh * h)))))
        return sorted(dets, key=lambda d: -d.score)


def _have_models() -> bool:
    try:
        import speciesnet  # noqa: F401
        return True
    except ImportError:
        return False


def load_detector() -> MegaDetector | None:
    if os.environ.get("TRAILWATCH_DETECTOR", "auto") == "off":
        return None
    if os.environ.get("TRAILWATCH_CLASSIFIER", "stub") != "speciesnet" or not _have_models():
        return None
    return MegaDetector()


def load_classifier() -> Classifier:
    if os.environ.get("TRAILWATCH_CLASSIFIER", "stub") == "speciesnet" and _have_models():
        return SpeciesNetClassifier()
    return StubClassifier()
