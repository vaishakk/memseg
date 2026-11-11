"""Result sinks: where segmentation results are written."""

from __future__ import annotations

import os

import numpy as np

from ..domain.models import ImageRef, SegmentationResult
from ..domain.ports import ResultSink
from .images import write_labels
from .previews import save_preview


def _out_path(out_dir: str, ref: ImageRef, suffix: str) -> str:
    return os.path.join(out_dir, os.path.splitext(ref.key)[0] + suffix)


class LabelTiffSink(ResultSink):
    """<out_dir>/<key without extension>.tif (uint16, uint32 if > 65535 instances)."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir

    def path(self, ref: ImageRef) -> str:
        return _out_path(self.out_dir, ref, ".tif")

    def exists(self, ref: ImageRef) -> bool:
        return os.path.isfile(self.path(ref))

    def save(self, ref: ImageRef, result: SegmentationResult) -> None:
        write_labels(self.path(ref), result.labels)


class PreviewSink(ResultSink):
    """Colour PNG preview; overlays (skeleton, junctions) are drawn if the method provides them."""

    def __init__(self, out_dir: str, suffix: str = "_preview.png", max_size: int = 1600, overlay: bool = True):
        self.out_dir, self.suffix, self.max_size, self.overlay = out_dir, suffix, max_size, overlay

    def exists(self, ref: ImageRef) -> bool:
        return os.path.isfile(_out_path(self.out_dir, ref, self.suffix))

    def save(self, ref: ImageRef, result: SegmentationResult) -> None:
        save_preview(_out_path(self.out_dir, ref, self.suffix), result.labels.data,
                     result.overlay if self.overlay else None, self.max_size)


class ArtifactSink(ResultSink):
    """Method-specific intermediate arrays (e.g. network outputs) as <key>_raw.npz, if the method produced any."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir

    def exists(self, ref: ImageRef) -> bool:
        return os.path.isfile(_out_path(self.out_dir, ref, "_raw.npz"))

    def save(self, ref: ImageRef, result: SegmentationResult) -> None:
        if not result.artifacts:
            return
        p = _out_path(self.out_dir, ref, "_raw.npz")
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        np.savez_compressed(p, **result.artifacts)
