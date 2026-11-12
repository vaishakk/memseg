"""Use cases. They orchestrate ports and know nothing about files, torch or a specific method."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..domain.metrics import match_instances, summarize
from ..domain.models import EvaluationSummary, ImageRef, ItemOutcome, Status
from ..domain.ports import (
    GroundTruthReader,
    ImageSource,
    InstanceSegmenter,
    MaskReader,
    NullSink,
    ProgressListener,
    ReportWriter,
    ResultSink,
)


def _describe_error(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


@dataclass
class SegmentImages:
    """Segment every image from a source and hand the results to a sink.

    Failures on single images are recorded and do not stop the run.
    """

    segmenter: InstanceSegmenter
    source: ImageSource
    reader: MaskReader
    sink: ResultSink = field(default_factory=NullSink)
    report: ReportWriter | None = None
    progress: ProgressListener = field(default_factory=ProgressListener)
    skip_existing: bool = True

    def run(self) -> list[ItemOutcome]:
        refs = list(self.source)
        self.progress.start(len(refs), f"segment with {self.segmenter.describe()}")
        outcomes = []
        for i, ref in enumerate(refs, 1):
            outcome = self._one(ref)
            outcomes.append(outcome)
            self.progress.item_done(i, len(refs), outcome)
        if self.report is not None:
            self.report.write(outcomes)
        self.progress.finished(outcomes, None)
        return outcomes

    def _one(self, ref: ImageRef) -> ItemOutcome:
        if self.skip_existing and self.sink.exists(ref):
            return ItemOutcome(ref=ref, status=Status.SKIPPED)
        t0 = time.time()
        try:
            mask = self.reader.read(ref)
            result = self.segmenter.segment(mask)
            self.sink.save(ref, result)
            return ItemOutcome(ref=ref, status=Status.OK, seconds=time.time() - t0, shape=mask.shape, result=result)
        except Exception as e:  # keep going on bad inputs
            return ItemOutcome(ref=ref, status=Status.FAILED, seconds=time.time() - t0, error=_describe_error(e))


@dataclass
class EvaluateSegmenter:
    """Segment images that have ground truth and score them (precision / recall / F1 / IoU).

    The sink is optional; give one to also keep the predicted label maps.
    """

    segmenter: InstanceSegmenter
    source: ImageSource
    reader: MaskReader
    gt_reader: GroundTruthReader
    sink: ResultSink = field(default_factory=NullSink)
    report: ReportWriter | None = None
    progress: ProgressListener = field(default_factory=ProgressListener)
    iou_threshold: float = 0.5

    def run(self) -> tuple[list[ItemOutcome], EvaluationSummary]:
        refs = list(self.source)
        self.progress.start(len(refs), f"evaluate {self.segmenter.describe()}")
        outcomes = []
        for i, ref in enumerate(refs, 1):
            t0 = time.time()
            try:
                if ref.gt_path is None:
                    raise ValueError("no ground truth for this image")
                mask = self.reader.read(ref)
                gt = self.gt_reader.read(ref)
                result = self.segmenter.segment(mask)
                self.sink.save(ref, result)
                m = match_instances(gt, result.labels, self.iou_threshold)
                outcome = ItemOutcome(
                    ref=ref, status=Status.OK, seconds=time.time() - t0, shape=mask.shape, result=result, match=m
                )
            except Exception as e:
                outcome = ItemOutcome(ref=ref, status=Status.FAILED, seconds=time.time() - t0, error=_describe_error(e))
            outcomes.append(outcome)
            self.progress.item_done(i, len(refs), outcome)
        summary = summarize([o.match for o in outcomes if o.match is not None])
        if self.report is not None:
            self.report.write(outcomes, summary)
        self.progress.finished(outcomes, summary)
        return outcomes, summary
