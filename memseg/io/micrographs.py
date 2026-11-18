"""File adapters for semantic segmentation: micrograph reading, mask/probability writing, progress."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from typing import TextIO

import numpy as np

from ..domain.models import (
    ImageRef,
    MembraneMask,
    Micrograph,
    MicrographOutcome,
    SegmentationResult,
    SemanticResult,
    SemanticSummary,
    Status,
)
from ..domain.ports import GroundTruthMaskReader, MicrographListener, MicrographReader, ResultSink, SemanticSink
from .images import read_image, to_binary

MICROGRAPH_EXTS = (".mrc", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def _mrcfile():
    try:
        import mrcfile
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise RuntimeError("reading .mrc files needs the 'mrcfile' package (pip install memseg[mrc])") from e
    return mrcfile


def header_pixel_size(path: str) -> float | None:
    """Pixel size (Å/px) from an MRC header, or None when absent or unset.

    0 means "not set"; 1.0 is what many programs write when they don't know it, so it is
    treated as unset too.
    """
    if not path.lower().endswith(".mrc"):
        return None
    try:
        with _mrcfile().open(path, header_only=True, permissive=True) as m:
            apix = float(m.voxel_size.x)
    except Exception:
        return None
    if not np.isfinite(apix) or apix <= 0 or abs(apix - 1.0) < 1e-6:
        return None
    return apix


def read_micrograph(path: str) -> np.ndarray:
    """2-D float32 image from .mrc / .tif / .png / ...; stacks give their first frame,
    colour images their mean channel."""
    if path.lower().endswith(".mrc"):
        mrcfile = _mrcfile()
        try:
            handle = mrcfile.mmap(path, permissive=True, mode="r")
        except Exception:
            handle = mrcfile.open(path, permissive=True)
        with handle as mrc:
            data = mrc.data[0] if mrc.data.ndim == 3 else mrc.data
            return np.array(data, dtype=np.float32, copy=True)
    a = np.asarray(read_image(path))
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = a[..., :3].mean(axis=-1)
    elif a.ndim == 3:
        a = a[0]
    return a.astype(np.float32, copy=False)


class MicrographFileReader(MicrographReader):
    """Reads micrographs; the pixel size comes from the MRC header (if `use_header` and set),
    else `default_pixel_size` (may be None)."""

    def __init__(self, default_pixel_size: float | None = None, use_header: bool = True):
        self.default_pixel_size, self.use_header = default_pixel_size, use_header

    def read(self, ref: ImageRef) -> Micrograph:
        apix = header_pixel_size(ref.path) if self.use_header else None
        return Micrograph(read_micrograph(ref.path), ref, apix if apix is not None else self.default_pixel_size)


class ImageFileGroundTruthMaskReader(GroundTruthMaskReader):
    def read(self, ref: ImageRef) -> MembraneMask:
        if ref.gt_path is None:
            raise ValueError(f"{ref.key}: no ground-truth path")
        return MembraneMask(to_binary(read_image(ref.gt_path)), ref)


def _out_path(out_dir: str, ref: ImageRef, suffix: str) -> str:
    return os.path.join(out_dir, os.path.splitext(ref.key)[0] + suffix)


class SemanticMaskSink(SemanticSink):
    """<out_dir>/<key>_mask.tif (uint8, 255 = membrane); with `probability`, also
    <key>_prob.tif (float32) when the method produced one."""

    def __init__(self, out_dir: str, probability: bool = False):
        self.out_dir, self.probability = out_dir, probability

    def path(self, ref: ImageRef) -> str:
        return _out_path(self.out_dir, ref, "_mask.tif")

    def exists(self, ref: ImageRef) -> bool:
        return os.path.isfile(self.path(ref))

    def save(self, ref: ImageRef, result: SemanticResult) -> None:
        import tifffile

        p = self.path(ref)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        tifffile.imwrite(p, result.mask.data.astype(np.uint8) * 255, compression="zlib")
        if self.probability and result.probability is not None:
            tifffile.imwrite(
                _out_path(self.out_dir, ref, "_prob.tif"),
                np.asarray(result.probability, np.float32),
                compression="zlib",
            )


class MaskedMicrographSink(SemanticSink, ResultSink):
    """<out_dir>/<key>_masked.mrc: the micrograph with everything outside the mask set to the
    micrograph's mean (float32,
    the pixel size of `reader` in the header).

    Works as a semantic sink (the membrane mask) and as an instance sink (the pixels of the kept
    instances). The micrograph is read again with `reader`, so the values and pixel size are the
    ones the method saw.
    """

    def __init__(self, out_dir: str, reader: MicrographReader, suffix: str = "_masked.mrc"):
        self.out_dir, self.reader, self.suffix = out_dir, reader, suffix

    def exists(self, ref: ImageRef) -> bool:
        return os.path.isfile(_out_path(self.out_dir, ref, self.suffix))

    def save(self, ref: ImageRef, result: SemanticResult | SegmentationResult) -> None:
        keep = result.mask.data if isinstance(result, SemanticResult) else result.labels.data > 0
        image = self.reader.read(ref)
        if keep.shape != image.shape:
            raise ValueError(f"{ref.key}: {keep.shape} mask for a {image.shape} micrograph")
        data = np.asarray(image.data, dtype=np.float32)
        masked = np.where(keep, data, np.float32(data.mean(dtype=np.float64)))
        p = _out_path(self.out_dir, ref, self.suffix)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with _mrcfile().new(p, overwrite=True) as m:
            m.set_data(masked)
            if image.pixel_size is not None:
                m.voxel_size = image.pixel_size


class SemanticPreviewSink(SemanticSink):
    """<out_dir>/<key><suffix>: the mask as a white-on-black PNG (downscaled, thin lines kept)."""

    def __init__(self, out_dir: str, suffix: str = "_mask.png", max_size: int = 1600):
        self.out_dir, self.suffix, self.max_size = out_dir, suffix, max_size

    def exists(self, ref: ImageRef) -> bool:
        return os.path.isfile(_out_path(self.out_dir, ref, self.suffix))

    def save(self, ref: ImageRef, result: SemanticResult) -> None:
        from PIL import Image

        from .previews import render

        p = _out_path(self.out_dir, ref, self.suffix)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        Image.fromarray(render(result.mask.data.astype(np.int32), max_size=self.max_size, mono=True)).save(p)


def _fmt(v) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


class ConsoleMicrographProgress(MicrographListener):
    def __init__(self, stream: TextIO | None = None, header: str = ""):
        self.stream, self.header = stream, header  # None: the current sys.stdout

    def _p(self, s: str) -> None:
        print(s, file=self.stream or sys.stdout, flush=True)

    def start(self, total: int, description: str) -> None:
        self._p("=" * 64)
        if self.header:
            self._p(self.header)
        self._p(f"{description}: {total} image(s)")
        self._p("=" * 64)

    def item_done(self, index: int, total: int, o: MicrographOutcome) -> None:
        head = f"[{index}/{total}] {o.ref.key}:"
        if o.status is Status.SKIPPED:
            self._p(f"{head} output exists, skipped")
            return
        if o.status is Status.FAILED:
            self._p(f"{head} FAILED  {o.error}")
            return
        s = o.semantic
        assert s is not None and o.shape is not None  # OK outcomes carry a result
        line = f"{head} {o.shape[1]}x{o.shape[0]}  membrane {100 * float(s.mask.data.mean()):.2f}%"
        if s.info:
            line += "  " + "  ".join(f"{k} {_fmt(v)}" for k, v in s.info.items())
        if o.instances is not None:
            line += f"  instances {o.instances.labels.n_instances}"
        if o.scores is not None:
            line += f"  IoU {o.scores.iou:.3f}  Dice {o.scores.dice:.3f}"
        self._p(line + f"  {o.seconds:.1f}s")

    def finished(self, outcomes: Sequence[MicrographOutcome], summary: SemanticSummary | None) -> None:
        n = {s: sum(o.status is s for o in outcomes) for s in Status}
        self._p(f"done: {n[Status.OK]} ok, {n[Status.FAILED]} failed, {n[Status.SKIPPED]} skipped")
        if summary is not None:
            self._p(
                "OVERALL: "
                + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.as_dict().items()})
            )


__all__ = [
    "MICROGRAPH_EXTS",
    "ConsoleMicrographProgress",
    "ImageFileGroundTruthMaskReader",
    "MaskedMicrographSink",
    "MicrographFileReader",
    "SemanticMaskSink",
    "SemanticPreviewSink",
    "header_pixel_size",
    "read_micrograph",
]
