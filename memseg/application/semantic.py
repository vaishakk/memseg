"""Semantic segmentation use case: micrograph -> membrane mask [-> instances].

Knows nothing about files, networks or a specific method.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..domain.metrics import pixel_scores, summarize_semantic
from ..domain.models import ImageRef, MicrographOutcome, SemanticSummary, Status
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


def _describe_error(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


@dataclass
class SegmentMicrographs:
    """For every micrograph: semantic segmentation, then optionally an instance method on its mask.

    * `instance`: any InstanceSegmenter; it receives the semantic mask (with the micrograph's ref).
    * `gt_reader`: when given, images whose `ref.gt_path` is set are scored against that
      ground-truth membrane mask (pixel IoU / Dice); a summary over the scored images is returned.
    * `semantic_sink` decides "already done" for `skip_existing`.

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

    def describe(self) -> str:
        d = self.semantic.describe()
        return d + (f" -> {self.instance.describe()}" if self.instance is not None else "")

    def run(self) -> tuple[list[MicrographOutcome], SemanticSummary | None]:
        refs = list(self.source)
        self.progress.start(len(refs), f"segment micrographs with {self.describe()}")
        outcomes = []
        for i, ref in enumerate(refs, 1):
            outcome = self._one(ref)
            outcomes.append(outcome)
            self.progress.item_done(i, len(refs), outcome)
        scored = [o.scores for o in outcomes if o.scores is not None]
        summary = summarize_semantic(scored) if scored else None
        if self.report is not None:
            self.report.write(outcomes, summary)
        self.progress.finished(outcomes, summary)
        return outcomes, summary

    def _one(self, ref: ImageRef) -> MicrographOutcome:
        if self.skip_existing and self.semantic_sink.exists(ref):
            return MicrographOutcome(ref=ref, status=Status.SKIPPED)
        t0 = time.time()
        try:
            image = self.reader.read(ref)
            sem = self.semantic.segment(image)
            if sem.mask.shape != image.shape:
                raise ValueError(f"method returned a {sem.mask.shape} mask for a {image.shape} image")
            sem.mask.ref = ref
            self.semantic_sink.save(ref, sem)
            inst = None
            if self.instance is not None:
                inst = self.instance.segment(sem.mask)
                self.instance_sink.save(ref, inst)
            scores = None
            if self.gt_reader is not None and ref.gt_path is not None:
                scores = pixel_scores(self.gt_reader.read(ref), sem.mask)
            return MicrographOutcome(
                ref=ref,
                status=Status.OK,
                seconds=time.time() - t0,
                shape=image.shape,
                semantic=sem,
                instances=inst,
                scores=scores,
            )
        except Exception as e:  # keep going on bad inputs
            return MicrographOutcome(ref=ref, status=Status.FAILED, seconds=time.time() - t0, error=_describe_error(e))
