"""Species classifiers behind one small interface.

SpeciesNet (Google's open camera-trap model) is used when installed; a
deterministic stub keeps tests and the demo fast and offline.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Prediction:
    label: str          # common name, or "blank" / "human" / "vehicle"
    scientific: str
    score: float        # classifier score in [0, 1]; not a calibrated probability
    model: str


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
        if r > b + 25 and r > g + 25:
            return Prediction("Asian elephant", "Elephas maximus", round(0.95 * q, 3), self.name)
        if b > r + 25 and b > g + 25:
            return Prediction("Spotted deer", "Axis axis", round(0.92 * q, 3), self.name)
        return Prediction("unknown animal", "", round(0.3 * q, 3), self.name)


class SpeciesNetClassifier(Classifier):
    name = "speciesnet"

    def __init__(self, model: str | None = None, country: str = "IND"):
        from speciesnet import SpeciesNet  # heavy import, only when used

        self.model_name = model or os.environ.get("SPECIESNET_MODEL", "kaggle:google/speciesnet/pyTorch/v4.0.2a/1")
        self.country = country
        self.model = SpeciesNet(self.model_name, components="classifier")
        self.name = f"speciesnet:{self.model_name.split('/')[-2] if '/' in self.model_name else self.model_name}"

    def classify(self, crop: np.ndarray) -> Prediction:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "crop.jpg")
            cv2.imwrite(p, crop)
            out = self.model.classify(filepaths=[p], country=self.country)
        pred = (out.get("predictions") or [{}])[0]
        classes = (pred.get("classifications") or {}).get("classes") or [""]
        scores = (pred.get("classifications") or {}).get("scores") or [0.0]
        # SpeciesNet labels look like "uuid;class;order;family;genus;species;common name"
        parts = classes[0].split(";")
        common = parts[-1] if parts and parts[-1] else "unknown"
        sci = " ".join(p for p in parts[4:6] if p).strip()
        return Prediction(common, sci, float(scores[0]), self.name)


def load_classifier() -> Classifier:
    if os.environ.get("TRAILWATCH_CLASSIFIER", "stub") == "speciesnet":
        return SpeciesNetClassifier()
    return StubClassifier()
