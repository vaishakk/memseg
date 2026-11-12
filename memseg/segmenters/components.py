"""Reference method: every connected piece of membrane is one instance.

Deliberately simple. It shows the method contract and gives the framework a method that
always works out of the box (no network, numpy/scipy only).
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from scipy import ndimage

from ..domain.models import InstanceLabels, MembraneMask, SegmentationResult
from ..domain.ports import InstanceSegmenter
from .base import SegmenterPlugin


class ConnectedComponentsSegmenter(InstanceSegmenter):
    name = "components"

    def __init__(self, min_size: int = 50):
        self.min_size = int(min_size)

    def describe(self) -> str:
        return f"components (min_size={self.min_size})"

    def segment(self, mask: MembraneMask) -> SegmentationResult:
        labels, n = ndimage.label(mask.data, structure=np.ones((3, 3)))
        if n:
            sizes = np.bincount(labels.ravel())
            labels[sizes[labels] < self.min_size] = 0
        return SegmentationResult(
            labels=InstanceLabels.consecutive(labels), method=self.name, info=dict(components=int(n))
        )


class ConnectedComponentsPlugin(SegmenterPlugin):
    name = "components"
    summary = "one instance per connected membrane piece (reference method)"

    def add_arguments(self, group: argparse._ArgumentGroup) -> None:
        group.add_argument("--min_size", type=int, default=50, help="drop pieces smaller than this (px)")

    def build(self, args: argparse.Namespace, overrides: dict[str, Any]) -> InstanceSegmenter:
        unknown = set(overrides) - {"min_size"}
        if unknown:
            raise ValueError(f"unknown components parameter(s): {sorted(unknown)}")
        return ConnectedComponentsSegmenter(overrides.get("min_size", getattr(args, "min_size", 50)))
