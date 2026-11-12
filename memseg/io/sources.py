"""Image sources: which images to process."""

from __future__ import annotations

import glob
import os
import random
from collections.abc import Iterator, Sequence

from ..domain.models import ImageRef
from ..domain.ports import ImageSource

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def _select(refs: list[ImageRef], sample: int | None, seed: int, limit: int | None) -> list[ImageRef]:
    if sample:
        refs = random.Random(seed).sample(refs, min(sample, len(refs)))
    if limit:
        refs = refs[:limit]
    return refs


class FolderImageSource(ImageSource):
    """Mask files given directly or found in folders (recursively by default).

    The key of an image is its path relative to the folder it was found in, or its file
    name if given directly; outputs mirror these keys.
    """

    def __init__(
        self,
        inputs: Sequence[str],
        exts: Sequence[str] = IMAGE_EXTS,
        recursive: bool = True,
        exclude_dirs: Sequence[str] = (),
        sample: int | None = None,
        seed: int = 0,
        limit: int | None = None,
        gt_dir: str | None = None,
    ):
        self.inputs = list(inputs)
        self.gt_dir = gt_dir
        self.exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in exts)
        self.recursive = recursive
        self.exclude = {os.path.abspath(d) for d in exclude_dirs}
        self.sample, self.seed, self.limit = sample, seed, limit

    def _scan(self) -> list[ImageRef]:
        refs = []
        for inp in self.inputs:
            inp = os.path.abspath(inp)
            if os.path.isfile(inp):
                refs.append(ImageRef(inp, os.path.basename(inp)))
                continue
            if not os.path.isdir(inp):
                raise FileNotFoundError(inp)
            for root, dirs, files in os.walk(inp):
                dirs[:] = sorted(
                    d
                    for d in dirs
                    if not d.startswith(".") and os.path.abspath(os.path.join(root, d)) not in self.exclude
                )
                for fn in sorted(files):
                    if not fn.startswith(".") and fn.lower().endswith(self.exts):
                        p = os.path.join(root, fn)
                        refs.append(ImageRef(p, os.path.relpath(p, inp)))
                if not self.recursive:
                    break
        return refs

    def _with_gt(self, ref: ImageRef) -> ImageRef:
        """Ground truth in gt_dir under the same relative path (any image extension), also
        accepting the mask_<tag> -> label_<tag> naming of generated datasets."""
        if not self.gt_dir:
            return ref
        stem = os.path.splitext(ref.key)[0]
        d, name = os.path.split(stem)
        for n in dict.fromkeys((name, name.replace("mask_", "label_", 1))):
            for ext in IMAGE_EXTS:
                p = os.path.join(self.gt_dir, d, n + ext)
                if os.path.isfile(p):
                    return ImageRef(ref.path, ref.key, p)
        return ref

    def __iter__(self) -> Iterator[ImageRef]:
        return iter(_select([self._with_gt(r) for r in self._scan()], self.sample, self.seed, self.limit))


def is_labelled_dataset(path: str) -> bool:
    return os.path.isdir(os.path.join(path, "masks")) and os.path.isdir(os.path.join(path, "labels"))


class LabelledDatasetSource(ImageSource):
    """Dataset folders laid out as <root>/masks/mask_<tag>.tif and <root>/labels/label_<tag>.tif
    (mask_<tag> in masks/ pairs with label_<tag> in labels/)."""

    def __init__(self, roots: Sequence[str], sample: int | None = None, seed: int = 0, limit: int | None = None):
        self.roots = list(roots)
        self.sample, self.seed, self.limit = sample, seed, limit

    def __iter__(self) -> Iterator[ImageRef]:
        refs = []
        multi = len(self.roots) > 1
        for root in self.roots:
            for p in sorted(glob.glob(os.path.join(root, "masks", "*.tif"))):
                fn = os.path.basename(p)
                gt = os.path.join(root, "labels", fn.replace("mask_", "label_", 1))
                key = os.path.join(os.path.basename(os.path.normpath(root)), fn) if multi else fn
                refs.append(ImageRef(p, key, gt if os.path.isfile(gt) else None))
        return iter(_select(refs, self.sample, self.seed, self.limit))
