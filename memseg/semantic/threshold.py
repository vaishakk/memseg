"""Reference semantic method: background-corrected intensity threshold.

Membranes in cryo-EM micrographs are dark lines. The image is flattened (a heavily blurred
copy is subtracted), lightly smoothed and robustly standardised (median / MAD); pixels darker
than `k` standard deviations become membrane, and pieces smaller than `min_area` are dropped.
Simple and fast: it gives the framework a semantic method that works without a network.
Defaults were tuned on one 5760x4092 micrograph at 1.74 Å/px against a U-Net mask (pixel IoU ~0.35):
useful as a baseline, not as a replacement for a trained model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from scipy import ndimage

from ..domain.models import MembraneMask, Micrograph, SemanticResult
from ..domain.ports import SemanticSegmenter
from .base import SemanticSegmenterPlugin


@dataclass
class ThresholdConfig:
    background_sigma: float = 50.0   # px; blur used as the background estimate
    smooth_sigma: float = 8.0        # px; noise suppression before thresholding
    k: float = 0.75                  # threshold in robust standard deviations
    polarity: int = -1               # -1: membranes darker than background, +1: brighter
    min_area: int = 2000             # px; drop smaller connected pieces

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any]) -> "ThresholdConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(overrides) - names
        if unknown:
            raise ValueError(f"unknown threshold parameter(s): {sorted(unknown)}; valid: {sorted(names)}")
        return cls(**overrides)


def _background(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian background, computed on a downsampled copy for speed."""
    f = max(1, int(sigma // 8))
    if f == 1:
        return ndimage.gaussian_filter(img, sigma)
    small = img[: img.shape[0] // f * f, : img.shape[1] // f * f]
    small = small.reshape(small.shape[0] // f, f, small.shape[1] // f, f).mean((1, 3))
    bg = ndimage.gaussian_filter(small, sigma / f, mode="reflect")
    zoom = (img.shape[0] / bg.shape[0], img.shape[1] / bg.shape[1])
    return ndimage.zoom(bg, zoom, order=1, mode="nearest")[: img.shape[0], : img.shape[1]]


class ThresholdSegmenter(SemanticSegmenter):
    name = "threshold"

    def __init__(self, config: ThresholdConfig | None = None):
        self.config = config or ThresholdConfig()

    def describe(self) -> str:
        c = self.config
        return f"threshold (k={c.k}, smooth={c.smooth_sigma}, min_area={c.min_area})"

    def segment(self, image: Micrograph) -> SemanticResult:
        c = self.config
        img = image.data
        flat = img - _background(img, c.background_sigma)
        if c.smooth_sigma > 0:
            flat = ndimage.gaussian_filter(flat, c.smooth_sigma)
        med = float(np.median(flat))
        mad = float(np.median(np.abs(flat - med))) * 1.4826 + 1e-12
        score = c.polarity * (flat - med) / mad
        mask = score > c.k
        n = 0
        if c.min_area > 1 and mask.any():
            lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
            sizes = np.bincount(lab.ravel())
            keep = sizes >= c.min_area
            keep[0] = False
            mask = keep[lab]
        prob = (1.0 / (1.0 + np.exp(-(score - c.k)))).astype(np.float32)
        return SemanticResult(mask=MembraneMask(mask, image.ref), method=self.name, probability=prob,
                              info=dict(pieces=int(n)))


class ThresholdPlugin(SemanticSegmenterPlugin):
    name = "threshold"
    summary = "background-corrected dark-line threshold (reference method, no network)"

    def build(self, args: argparse.Namespace, overrides: dict[str, Any]) -> SemanticSegmenter:
        return ThresholdSegmenter(ThresholdConfig.from_overrides(overrides))
