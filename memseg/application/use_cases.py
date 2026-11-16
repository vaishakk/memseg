"""Use cases. They orchestrate ports and know nothing about files, torch or a specific method."""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass, field

from ..domain.metrics import summarize
from ..domain.models import EvaluationSummary, ItemOutcome
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
from .parallel import ImageJob, run_images


@dataclass
class SegmentImages:
    """Segment every image from a source and hand the results to a sink.

    Failures on single images are recorded and do not stop the run. With an `executor`
    (injected by the composition root), images are processed in parallel; see parallel.py.
    """

    segmenter: InstanceSegmenter
    source: ImageSource
    reader: MaskReader
    sink: ResultSink = field(default_factory=NullSink)
    report: ReportWriter | None = None
    progress: ProgressListener = field(default_factory=ProgressListener)
    skip_existing: bool = True
    executor: Executor | None = None
    batch_size: int | None = None  # None: the staged segmenter's preferred batch
    max_in_flight: int = 8  # images read but not finished (parallel runs)

    def run(self) -> list[ItemOutcome]:
        refs = list(self.source)
        self.progress.start(len(refs), f"segment with {self.segmenter.describe()}")
        count = 0

        def on_done(o: ItemOutcome) -> None:
            nonlocal count
            count += 1
            self.progress.item_done(count, len(refs), o)

        outcomes = run_images(
            refs,
            self.segmenter,
            ImageJob(self.reader, self.sink),
            on_done,
            skip=self.sink.exists if self.skip_existing else None,
            executor=self.executor,
            batch_size=self.batch_size,
            max_in_flight=self.max_in_flight,
        )
        if self.report is not None:
            self.report.write(outcomes)
        self.progress.finished(outcomes, None)
        return outcomes


@dataclass
class EvaluateSegmenter:
    """Segment images that have ground truth and score them (precision / recall / F1 / IoU).

    The sink is optional; give one to also keep the predicted label maps. With an `executor`,
    images are processed in parallel; see parallel.py.
    """

    segmenter: InstanceSegmenter
    source: ImageSource
    reader: MaskReader
    gt_reader: GroundTruthReader
    sink: ResultSink = field(default_factory=NullSink)
    report: ReportWriter | None = None
    progress: ProgressListener = field(default_factory=ProgressListener)
    iou_threshold: float = 0.5
    executor: Executor | None = None
    batch_size: int | None = None
    max_in_flight: int = 8

    def run(self) -> tuple[list[ItemOutcome], EvaluationSummary]:
        refs = list(self.source)
        self.progress.start(len(refs), f"evaluate {self.segmenter.describe()}")
        count = 0

        def on_done(o: ItemOutcome) -> None:
            nonlocal count
            count += 1
            self.progress.item_done(count, len(refs), o)

        outcomes = run_images(
            refs,
            self.segmenter,
            ImageJob(self.reader, self.sink, self.gt_reader, self.iou_threshold),
            on_done,
            executor=self.executor,
            batch_size=self.batch_size,
            max_in_flight=self.max_in_flight,
        )
        summary = summarize([o.match for o in outcomes if o.match is not None])
        if self.report is not None:
            self.report.write(outcomes, summary)
        self.progress.finished(outcomes, summary)
        return outcomes, summary
