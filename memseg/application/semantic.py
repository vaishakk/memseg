"""Semantic segmentation use case: micrograph -> membrane mask [-> instances].

Knows nothing about files, networks or a specific method.
"""

from __future__ import annotations

from concurrent.futures import Executor
from dataclasses import dataclass, field

from ..domain.metrics import summarize_semantic
from ..domain.models import MicrographOutcome, SemanticSummary
from ..domain.ports import (
    GroundTruthMaskReader,
    ImageSource,
    InstanceSegmenter,
    MicrographListener,
    MicrographReader,
    NullSink,
    ResultSink,
    SemanticReportWriter,
    SemanticSegmenter,
    SemanticSink,
)
from .parallel import MicrographJob, run_images


@dataclass
class SegmentMicrographs:
    """For every micrograph: semantic segmentation, then optionally an instance method on its mask.

    * `instance`: any InstanceSegmenter; it receives the semantic mask (with the micrograph's ref).
    * `gt_reader`: when given, images whose `ref.gt_path` is set are scored against that
      ground-truth membrane mask (pixel IoU / Dice); a summary over the scored images is returned.
    * `semantic_sink` decides "already done" for `skip_existing`.
    * `executor`: process micrographs in parallel (see parallel.py). A staged semantic method keeps
      `infer` in this process; its CPU stages, the instance method, saving and scoring run in workers.

    Failures on single images are recorded and do not stop the run.
    """

    semantic: SemanticSegmenter
    source: ImageSource
    reader: MicrographReader
    instance: InstanceSegmenter | None = None
    semantic_sink: SemanticSink = field(default_factory=NullSink)
    instance_sink: ResultSink = field(default_factory=NullSink)
    gt_reader: GroundTruthMaskReader | None = None
    report: SemanticReportWriter | None = None
    progress: MicrographListener = field(default_factory=MicrographListener)
    skip_existing: bool = False
    executor: Executor | None = None
    batch_size: int | None = None  # None: the staged semantic method's preferred batch
    max_in_flight: int = 8  # micrographs read but not finished (parallel runs)

    def describe(self) -> str:
        d = self.semantic.describe()
        return d + (f" -> {self.instance.describe()}" if self.instance is not None else "")

    def run(self) -> tuple[list[MicrographOutcome], SemanticSummary | None]:
        refs = list(self.source)
        self.progress.start(len(refs), f"segment micrographs with {self.describe()}")
        count = 0

        def on_done(o: MicrographOutcome) -> None:
            nonlocal count
            count += 1
            self.progress.item_done(count, len(refs), o)

        job = MicrographJob(self.reader, self.semantic_sink, self.instance, self.instance_sink, self.gt_reader)
        outcomes = run_images(
            refs,
            self.semantic,
            job,
            on_done,
            skip=self.semantic_sink.exists if self.skip_existing else None,
            executor=self.executor,
            batch_size=self.batch_size,
            max_in_flight=self.max_in_flight,
        )
        scored = [o.scores for o in outcomes if o.scores is not None]
        summary = summarize_semantic(scored) if scored else None
        if self.report is not None:
            self.report.write(outcomes, summary)
        self.progress.finished(outcomes, summary)
        return outcomes, summary
