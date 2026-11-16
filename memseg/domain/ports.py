"""Ports: interfaces the application layer depends on. Adapters implement them.

The segmentation methods are the main inverted dependencies: use cases only know
`InstanceSegmenter` (mask -> instances) and `SemanticSegmenter` (micrograph -> mask);
concrete methods are plugins found via memseg.segmenters / memseg.semantic and are
injected by the composition root (memseg.cli).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from .models import (
    ComparisonItem,
    ComparisonSummary,
    EvaluationSummary,
    ImageRef,
    InstanceLabels,
    ItemOutcome,
    MembraneMask,
    Micrograph,
    MicrographOutcome,
    SegmentationResult,
    SemanticResult,
    SemanticSummary,
)


class InstanceSegmenter(ABC):
    """A membrane instance segmentation method: binary mask in, instance labels out."""

    #: short registry name, e.g. "components"
    name: str = "unnamed"

    @abstractmethod
    def segment(self, mask: MembraneMask) -> SegmentationResult:
        """Segment one mask. Must not perform any file I/O."""

    def describe(self) -> str:
        """One line for logs (model, key parameters)."""
        return self.name


class SegmenterStages(ABC):
    """The per-image CPU stages of a `StagedSegmenter`.

    Use cases may run them in worker processes, so an instance must be picklable: plain
    configuration only (no model, no open files, no GPU state), and its module must not
    import heavy GPU libraries at import time.
    """

    @abstractmethod
    def prepare(self, mask: MembraneMask) -> Any:
        """Per-image work before `StagedSegmenter.infer` (e.g. rescaling). Result must be picklable."""

    @abstractmethod
    def finish(self, mask: MembraneMask, prepared: Any, inferred: Any) -> SegmentationResult:
        """Per-image work after `infer` (post-processing) -> the final result. No file I/O."""


class StagedSegmenter(InstanceSegmenter):
    """An instance method split into per-image CPU stages and one batchable stage.

        prepare(mask)  ->  infer([prepared, ...])  ->  finish(mask, prepared, inferred)
        (any process)      (the owning process)        (any process)

    `infer` holds the expensive shared state (e.g. a network on a GPU) and receives batches;
    `prepare` / `finish` come from `stages()` and can run for many images at once in other
    processes. `segment` runs the three stages in order, so a sequential run and a parallel
    run execute the same code and give the same results.
    """

    #: preferred number of images per `infer` call
    batch_size: int = 1

    @abstractmethod
    def stages(self) -> SegmenterStages:
        """The picklable per-image stages (see SegmenterStages)."""

    @abstractmethod
    def infer(self, prepared: Sequence[Any]) -> list[Any]:
        """One result per prepared item, in order. Results must be picklable."""

    def segment(self, mask: MembraneMask) -> SegmentationResult:
        st = self.stages()
        prepared = st.prepare(mask)
        return st.finish(mask, prepared, self.infer([prepared])[0])


class ImageSource(ABC):
    """Enumerates the images a use case should process."""

    @abstractmethod
    def __iter__(self) -> Iterator[ImageRef]: ...


class MaskReader(ABC):
    @abstractmethod
    def read(self, ref: ImageRef) -> MembraneMask: ...


class GroundTruthReader(ABC):
    @abstractmethod
    def read(self, ref: ImageRef) -> InstanceLabels: ...


class ResultSink(ABC):
    """Persists results (label maps, previews, raw artifacts, ...)."""

    @abstractmethod
    def exists(self, ref: ImageRef) -> bool:
        """True if output for `ref` already exists (used to skip work)."""

    @abstractmethod
    def save(self, ref: ImageRef, result: SegmentationResult) -> None: ...


class ReportWriter(ABC):
    @abstractmethod
    def write(self, outcomes: Sequence[ItemOutcome], summary: EvaluationSummary | None = None) -> None: ...


class ProgressListener(ABC):
    """Receives progress events (console printer, GUI, logging...)."""

    def start(self, total: int, description: str) -> None:  # pragma: no cover - default no-op
        pass

    def item_done(self, index: int, total: int, outcome: ItemOutcome) -> None:  # pragma: no cover
        pass

    def finished(self, outcomes: Sequence[ItemOutcome], summary: EvaluationSummary | None) -> None:  # pragma: no cover
        pass


class ComparisonSink(ABC):
    """Persists one compared image (e.g. a side-by-side preview of all methods)."""

    @abstractmethod
    def save(self, item: ComparisonItem, mask: MembraneMask, gt: InstanceLabels | None) -> None: ...


class ComparisonReportWriter(ABC):
    @abstractmethod
    def write(self, items: Sequence[ComparisonItem], summary: ComparisonSummary) -> None: ...


class ComparisonListener(ABC):
    def start(self, total: int, methods: Sequence[str]) -> None:  # pragma: no cover - default no-op
        pass

    def item_done(self, index: int, total: int, item: ComparisonItem) -> None:  # pragma: no cover
        pass

    def finished(self, items: Sequence[ComparisonItem], summary: ComparisonSummary) -> None:  # pragma: no cover
        pass


# ───────────────────────────────────────────── semantic segmentation ──


class SemanticSegmenter(ABC):
    """A semantic segmentation method: micrograph in, membrane mask (+ optional probability) out."""

    #: short registry name, e.g. "threshold"
    name: str = "unnamed"

    @abstractmethod
    def segment(self, image: Micrograph) -> SemanticResult:
        """Segment one micrograph. Must not perform any file I/O."""

    def describe(self) -> str:
        """One line for logs (model, key parameters)."""
        return self.name


class SemanticStages(ABC):
    """The per-image CPU stages of a `StagedSemanticSegmenter`; same rules as `SegmenterStages`
    (picklable, configuration only, no GPU libraries imported at module level)."""

    @abstractmethod
    def prepare(self, image: Micrograph) -> Any:
        """Per-image work before `infer`. Result must be picklable."""

    @abstractmethod
    def finish(self, image: Micrograph, prepared: Any, inferred: Any) -> SemanticResult:
        """Per-image work after `infer` -> the final result. No file I/O."""


class StagedSemanticSegmenter(SemanticSegmenter):
    """A semantic method split like `StagedSegmenter`: prepare (any process) -> infer on batches
    (the process that holds the model) -> finish (any process)."""

    #: preferred number of images per `infer` call
    batch_size: int = 1

    @abstractmethod
    def stages(self) -> SemanticStages:
        """The picklable per-image stages."""

    @abstractmethod
    def infer(self, prepared: Sequence[Any]) -> list[Any]:
        """One result per prepared item, in order. Results must be picklable."""

    def segment(self, image: Micrograph) -> SemanticResult:
        st = self.stages()
        prepared = st.prepare(image)
        return st.finish(image, prepared, self.infer([prepared])[0])


class MicrographReader(ABC):
    @abstractmethod
    def read(self, ref: ImageRef) -> Micrograph: ...


class GroundTruthMaskReader(ABC):
    """Reads the ground-truth membrane mask for `ref.gt_path`."""

    @abstractmethod
    def read(self, ref: ImageRef) -> MembraneMask: ...


class SemanticSink(ABC):
    """Persists semantic results. `CompositeSink` and `NullSink` work with these sinks too."""

    def exists(self, ref: ImageRef) -> bool:
        return False

    @abstractmethod
    def save(self, ref: ImageRef, result: SemanticResult) -> None: ...


class NullSink(ResultSink, SemanticSink):
    """Discards results (instance or semantic)."""

    def exists(self, ref: ImageRef) -> bool:
        return False

    def save(self, ref: ImageRef, result: Any) -> None:
        pass


class CompositeSink(ResultSink, SemanticSink):
    """Fan out to several sinks of one family (all ResultSink or all SemanticSink).

    The first sink is the primary output: `exists` asks it only,
    so a missing optional extra (e.g. a preview) does not force recomputation."""

    def __init__(self, sinks: Iterable[ResultSink] | Iterable[SemanticSink]):
        self.sinks: list[ResultSink | SemanticSink] = list(sinks)

    def exists(self, ref: ImageRef) -> bool:
        return bool(self.sinks) and self.sinks[0].exists(ref)

    def save(self, ref: ImageRef, result: Any) -> None:
        for s in self.sinks:
            s.save(ref, result)


class SemanticReportWriter(ABC):
    @abstractmethod
    def write(self, outcomes: Sequence[MicrographOutcome], summary: SemanticSummary | None = None) -> None: ...


class MicrographListener(ABC):
    def start(self, total: int, description: str) -> None:  # pragma: no cover - default no-op
        pass

    def item_done(self, index: int, total: int, outcome: MicrographOutcome) -> None:  # pragma: no cover
        pass

    def finished(
        self, outcomes: Sequence[MicrographOutcome], summary: SemanticSummary | None
    ) -> None:  # pragma: no cover
        pass
