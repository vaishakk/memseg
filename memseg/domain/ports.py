"""Ports: interfaces the application layer depends on. Adapters implement them.

The segmentation methods are the main inverted dependencies: use cases only know
`InstanceSegmenter` (mask -> instances) and `SemanticSegmenter` (micrograph -> mask);
concrete methods are plugins found via memseg.segmenters / memseg.semantic and are
injected by the composition root (memseg.cli).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Optional, Sequence

from .models import (ComparisonItem, ComparisonSummary, EvaluationSummary, ImageRef, InstanceLabels, ItemOutcome,
                     MembraneMask, Micrograph, MicrographOutcome, SegmentationResult, SemanticResult, SemanticSummary)


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


class NullSink(ResultSink):
    def exists(self, ref: ImageRef) -> bool:
        return False

    def save(self, ref: ImageRef, result: SegmentationResult) -> None:
        pass


class CompositeSink(ResultSink):
    """Fan out to several sinks. The first sink is the primary output: `exists` asks it only,
    so a missing optional extra (e.g. a preview) does not force recomputation."""

    def __init__(self, sinks: Iterable[ResultSink]):
        self.sinks = list(sinks)

    def exists(self, ref: ImageRef) -> bool:
        return bool(self.sinks) and self.sinks[0].exists(ref)

    def save(self, ref: ImageRef, result: SegmentationResult) -> None:
        for s in self.sinks:
            s.save(ref, result)


class ComparisonSink(ABC):
    """Persists one compared image (e.g. a side-by-side preview of all methods)."""

    @abstractmethod
    def save(self, item: ComparisonItem, mask: MembraneMask, gt: Optional[InstanceLabels]) -> None: ...


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


class SemanticReportWriter(ABC):
    @abstractmethod
    def write(self, outcomes: Sequence[MicrographOutcome], summary: SemanticSummary | None = None) -> None: ...


class MicrographListener(ABC):
    def start(self, total: int, description: str) -> None:  # pragma: no cover - default no-op
        pass

    def item_done(self, index: int, total: int, outcome: MicrographOutcome) -> None:  # pragma: no cover
        pass

    def finished(self, outcomes: Sequence[MicrographOutcome], summary: SemanticSummary | None) -> None:  # pragma: no cover
        pass
