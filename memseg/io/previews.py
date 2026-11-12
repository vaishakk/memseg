"""Colour previews of instance label maps."""

from __future__ import annotations

import os

import numpy as np
from scipy import ndimage as ndi

from ..domain.models import Overlay


def palette(n: int) -> np.ndarray:
    """Well-separated colours (golden-ratio hues); index 0 is black."""
    from skimage.color import hsv2rgb

    i = np.arange(n + 1, dtype=np.float32)
    hsv = np.stack(
        [np.mod(i * 0.618034, 1), 0.75 + 0.25 * np.mod(i * 0.38197, 1), 0.8 + 0.2 * np.mod(i * 0.23607, 1)], -1
    )
    rgb = (hsv2rgb(hsv[None])[0] * 255).astype(np.uint8)
    rgb[0] = 0
    return rgb


def render(labels: np.ndarray, overlay: Overlay | None = None, max_size: int = 1600, mono: bool = False) -> np.ndarray:
    """Colour image of a label map; mono=True draws every non-zero pixel white (for input masks)."""
    if mono:
        rgb = np.zeros((*labels.shape, 3), np.uint8)
        rgb[labels > 0] = 235
    else:
        rgb = palette(int(labels.max(initial=0)))[labels]
    if overlay is not None:
        if overlay.junctions is not None:
            jz = ndi.binary_dilation(overlay.junctions, iterations=max(2, int(overlay.thickness)))
            rgb[jz & (labels == 0)] = (90, 90, 90)
            rgb[jz & (labels > 0)] //= 2
        if overlay.skeleton is not None:
            rgb[overlay.skeleton] = 255
    f = int(np.ceil(max(labels.shape) / max_size)) if max_size else 1
    if f > 1:  # max-pool on brightness so thin membranes survive the downscale
        H, W = labels.shape
        Hc, Wc = H // f * f, W // f * f
        r = rgb[:Hc, :Wc].reshape(Hc // f, f, Wc // f, f, 3)
        lum = r.sum(-1)
        pick = lum.reshape(Hc // f, f, Wc // f, f).transpose(0, 2, 1, 3).reshape(Hc // f, Wc // f, -1).argmax(-1)
        r = r.transpose(0, 2, 1, 3, 4).reshape(Hc // f, Wc // f, f * f, 3)
        rgb = np.take_along_axis(r, pick[..., None, None], 2)[:, :, 0]
    return rgb


def save_preview(path: str, labels: np.ndarray, overlay: Overlay | None = None, max_size: int = 1600) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(render(labels, overlay, max_size)).save(path)
