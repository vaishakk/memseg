"""Side-by-side comparison previews: input mask | ground truth (if any) | one panel per method."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..domain.models import ComparisonItem, InstanceLabels, MembraneMask, Status
from ..domain.ports import ComparisonSink
from .previews import render

BG = (25, 25, 25)
TEXT = (235, 235, 235)
DIM = (160, 160, 160)
BAD = (240, 110, 100)


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


class SideBySideSink(ComparisonSink):
    """<out_dir>/<key>.png with one captioned panel per image/method.

    Captions: method name, instances, F1 / pair-AUROC (when ground truth exists), run time.
    """

    def __init__(self, out_dir: str, panel_size: int = 900, gap: int = 8):
        self.out_dir, self.panel_size, self.gap = out_dir, panel_size, gap

    def path(self, item: ComparisonItem) -> str:
        return os.path.join(self.out_dir, os.path.splitext(item.ref.key)[0] + ".png")

    def save(self, item: ComparisonItem, mask: MembraneMask, gt: Optional[InstanceLabels]) -> None:
        from PIL import Image, ImageDraw
        panels: list[tuple[np.ndarray, list[tuple[str, tuple]]]] = []
        panels.append((render(mask.data.astype(np.int32), max_size=self.panel_size, mono=True),
                       [("input mask", TEXT), (f"{mask.width}x{mask.height}", DIM)]))
        if gt is not None:
            panels.append((render(gt.data, max_size=self.panel_size),
                           [("ground truth", TEXT), (f"{gt.n_instances} instances", DIM)]))
        size = panels[0][0].shape[:2]
        for name, o in item.outcomes.items():
            if o.status is Status.OK:
                img = render(o.result.labels.data, max_size=self.panel_size)
                line = f"{o.result.labels.n_instances} instances   {o.seconds:.2f} s"
                lines = [(name, TEXT), (line, DIM)]
                if o.match is not None:
                    auc = o.pairs.auroc if o.pairs is not None else None
                    lines.append((f"F1 {o.f1:.3f}   AUROC {auc:.3f}" if auc is not None else f"F1 {o.f1:.3f}", TEXT))
            else:
                img = np.full(size + (3,), 40, np.uint8)
                lines = [(name, TEXT), ("FAILED", BAD), ((o.error or "")[:60], DIM)]
            panels.append((img, lines))

        h, w = size
        fs = max(14, w // 28)
        cap_h = int(fs * 1.35) * 3 + fs // 2
        W = len(panels) * w + (len(panels) + 1) * self.gap
        H = h + cap_h + 2 * self.gap
        canvas = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(canvas)
        font = _font(fs)
        for k, (img, lines) in enumerate(panels):
            x = self.gap + k * (w + self.gap)
            y = self.gap
            for j, (txt, col) in enumerate(lines):
                draw.text((x + 4, y + j * int(fs * 1.35)), txt, fill=col, font=font)
            canvas.paste(Image.fromarray(img), (x, y + cap_h))
        os.makedirs(os.path.dirname(self.path(item)) or ".", exist_ok=True)
        canvas.save(self.path(item))
