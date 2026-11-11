"""Use case: run several segmentation methods on the same images and compare them."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Optional

from ..domain.metrics import match_instances, pair_counts, sample_pixel_pairs, summarize
from ..domain.models import (ComparisonItem, ComparisonSummary, ImageRef, InstanceLabels, MembraneMask,
                             MethodOutcome, MethodSummary, PairCounts, Status)
from ..domain.ports import (ComparisonListener, ComparisonReportWriter, ComparisonSink, GroundTruthReader,
                            ImageSource, InstanceSegmenter, MaskReader, ResultSink)


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


@dataclass
class PairSampling:
    """Pixel pairs used for the same-instance AUROC (identical pairs for every method)."""

    n_pairs: int = 20000
    radius: float = 100.0
    seed: int = 0


@dataclass
class CompareSegmenters:
    """Segment every image with each method, time it, and, when ground truth exists, score it.

    segmenters:     name -> method (order is kept in reports and previews)
    setup_seconds:  name -> time spent constructing the method (e.g. loading a network)
    method_sinks:   optional name -> sink for that method's label maps
    """

    segmenters: Mapping[str, InstanceSegmenter]
    source: ImageSource
    reader: MaskReader
    gt_reader: Optional[GroundTruthReader] = None
    method_sinks: Mapping[str, ResultSink] = field(default_factory=dict)
    comparison_sink: Optional[ComparisonSink] = None
    report: Optional[ComparisonReportWriter] = None
    progress: ComparisonListener = field(default_factory=ComparisonListener)
    setup_seconds: Mapping[str, float] = field(default_factory=dict)
    iou_threshold: float = 0.5
    pairs: PairSampling = field(default_factory=PairSampling)
    warmup: bool = False  # run each method once on the first image, untimed (GPU/cuDNN start-up)

    def run(self) -> tuple[list[ComparisonItem], ComparisonSummary]:
        if len(self.segmenters) < 2:
            raise ValueError("comparison needs at least two methods")
        refs = list(self.source)
        names = list(self.segmenters)
        self.progress.start(len(refs), names)
        if self.warmup and refs:
            self._warm(refs[0])
        items = []
        for i, ref in enumerate(refs, 1):
            item = self._one(ref, names)
            items.append(item)
            self.progress.item_done(i, len(refs), item)
        summary = self._summarize(items, names)
        if self.report is not None:
            self.report.write(items, summary)
        self.progress.finished(items, summary)
        return items, summary

    def _one(self, ref: ImageRef, names: list[str]) -> ComparisonItem:
        item = ComparisonItem(ref=ref)
        try:
            mask = self.reader.read(ref)
            gt = self._read_gt(ref)
        except Exception as e:
            item.error = _err(e)
            return item
        item.shape, item.has_gt = mask.shape, gt is not None
        pa = pb = None
        if gt is not None:
            item.n_gt = gt.n_instances
            pa, pb = sample_pixel_pairs(gt, self.pairs.n_pairs, self.pairs.radius, self.pairs.seed)
        for name in names:
            item.outcomes[name] = self._run_method(name, ref, mask, gt, pa, pb)
        if self.comparison_sink is not None:
            self.comparison_sink.save(item, mask, gt)
        return item

    def _warm(self, ref: ImageRef) -> None:
        try:
            mask = self.reader.read(ref)
        except Exception:
            return
        for seg in self.segmenters.values():
            try:
                seg.segment(mask)
            except Exception:
                pass  # the real run records the failure

    def _read_gt(self, ref: ImageRef) -> Optional[InstanceLabels]:
        if self.gt_reader is None or ref.gt_path is None:
            return None
        return self.gt_reader.read(ref)

    def _run_method(self, name, ref, mask: MembraneMask, gt, pa, pb) -> MethodOutcome:
        t0 = time.perf_counter()
        try:
            result = self.segmenters[name].segment(mask)
        except Exception as e:
            return MethodOutcome(name, Status.FAILED, time.perf_counter() - t0, error=_err(e))
        seconds = time.perf_counter() - t0  # segmentation only, excludes reading and writing
        out = MethodOutcome(name, Status.OK, seconds, result=result)
        if gt is not None:
            out.match = match_instances(gt, result.labels, self.iou_threshold)
            out.pairs = pair_counts(gt, result.labels, pa, pb)
        sink = self.method_sinks.get(name)
        if sink is not None:
            sink.save(ref, result)
        return out

    def _summarize(self, items: list[ComparisonItem], names: list[str]) -> ComparisonSummary:
        out = []
        for name in names:
            outs = [it.outcomes[name] for it in items if name in it.outcomes]
            ok = [o for o in outs if o.status is Status.OK]
            matches = [o.match for o in ok if o.match is not None]
            pairs = None
            if matches:
                pairs = PairCounts()
                for o in ok:
                    if o.pairs is not None:
                        pairs = pairs + o.pairs
            total = sum(o.seconds for o in ok)
            out.append(MethodSummary(
                method=name, description=self.segmenters[name].describe(), n_ok=len(ok),
                n_failed=len(outs) - len(ok), setup_seconds=self.setup_seconds.get(name, 0.0),
                total_seconds=total, mean_seconds=total / len(ok) if ok else 0.0,
                total_instances=sum(o.result.labels.n_instances for o in ok),
                evaluation=summarize(matches) if matches else None, pairs=pairs))
        return ComparisonSummary(methods=out, n_images=len(items), n_with_gt=sum(it.has_gt for it in items))
