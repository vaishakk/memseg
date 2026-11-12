"""Image file reading/writing adapters."""

from __future__ import annotations

import os

import numpy as np

from ..domain.models import ImageRef, InstanceLabels, MembraneMask
from ..domain.ports import GroundTruthReader, MaskReader


def read_image(path: str) -> np.ndarray:
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile

        return np.asarray(tifffile.imread(path))
    from PIL import Image

    return np.asarray(Image.open(path))


def to_binary(arr: np.ndarray) -> np.ndarray:
    """Any image -> 2D bool (RGB/RGBA reduced to the max channel, anything > 0 is membrane)."""
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].max(axis=-1)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {arr.shape}")
    return arr > 0


def write_labels(path: str, labels: InstanceLabels) -> None:
    import tifffile

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    a = labels.data
    dtype = np.uint16 if a.max(initial=0) <= np.iinfo(np.uint16).max else np.uint32
    tifffile.imwrite(path, a.astype(dtype), compression="zlib")


class ImageFileMaskReader(MaskReader):
    def read(self, ref: ImageRef) -> MembraneMask:
        return MembraneMask(to_binary(read_image(ref.path)), ref)


class ImageFileGroundTruthReader(GroundTruthReader):
    def read(self, ref: ImageRef) -> InstanceLabels:
        if ref.gt_path is None:
            raise ValueError(f"{ref.key}: no ground-truth path")
        return InstanceLabels(read_image(ref.gt_path).astype(np.int32))
