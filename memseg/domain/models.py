"""Domain models. Plain data, no I/O, no framework dependencies (numpy only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class ImageRef:
    """Identifies one input image.

    path:     where the mask can be read
    key:      stable relative identifier (used to name outputs, e.g. "sub/mask_001.tif")
    gt_path:  optional ground-truth instance label image for the same mask
    """

    path: str
    key: str
    gt_path: Optional[str] = None


@dataclass
class MembraneMask:
    """Binary membrane mask: True on membrane pixels."""

    data: np.ndarray
    ref: Optional[ImageRef] = None

    def __post_init__(self):
        a = np.asarray(self.data)
        if a.ndim != 2:
            raise ValueError(f"membrane mask must be 2D, got shape {a.shape}")
        self.data = a.astype(bool, copy=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]


@dataclass
class InstanceLabels:
    """Instance label map: 0 = background, 1..N = instances."""

    data: np.ndarray

    def __post_init__(self):
        a = np.asarray(self.data)
        if a.ndim != 2:
            raise ValueError(f"label map must be 2D, got shape {a.shape}")
        self.data = a.astype(np.int32, copy=False)

    @property
    def n_instances(self) -> int:
        return int(self.data.max()) if self.data.size else 0

    @classmethod
    def consecutive(cls, data: np.ndarray) -> "InstanceLabels":
        """Relabel any integer map to consecutive ids 1..N (negative/zero -> background)."""
        a = np.asarray(data).astype(np.int64)
        a[a < 0] = 0
        ids = np.unique(a)
        ids = ids[ids > 0]
        lut = np.zeros(int(a.max()) + 1 if a.size else 1, np.int32)
        lut[ids] = np.arange(1, len(ids) + 1)
        return cls(lut[a])


@dataclass
class Overlay:
    """Optional visual diagnostics a method can attach (drawn on previews)."""

    skeleton: Optional[np.ndarray] = None   # bool (H, W)
    junctions: Optional[np.ndarray] = None  # bool (H, W)
    thickness: float = 0.0


@dataclass
class SegmentationResult:
    """Output of one segmentation method on one mask.

    info:       small scalar diagnostics (written to reports), e.g. {"scale": 2}
    artifacts:  optional large intermediate arrays (e.g. network outputs) that a sink may save
    overlay:    optional drawing hints for previews
    """

    labels: InstanceLabels
    method: str
    info: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, np.ndarray] = field(default_factory=dict)
    overlay: Optional[Overlay] = None


@dataclass(frozen=True)
class MatchResult:
    """Comparison of a prediction with ground truth for one image."""

    tp: int
    fp: int
    fn: int
    ious: tuple[float, ...]
    matched_gt: frozenset[int]


@dataclass(frozen=True)
class EvaluationSummary:
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    n_images: int
    recall_touching: Optional[float] = None

    def as_dict(self) -> dict[str, float]:
        d = dict(precision=self.precision, recall=self.recall, f1=self.f1,
                 mean_matched_iou=self.mean_matched_iou, n_images=self.n_images)
        if self.recall_touching is not None:
            d["recall_touching"] = self.recall_touching
        return d


class Status(str, Enum):
    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class ItemOutcome:
    """What happened to one image in a use case run."""

    ref: ImageRef
    status: Status
    seconds: float = 0.0
    shape: Optional[tuple[int, int]] = None
    result: Optional[SegmentationResult] = None
    match: Optional[MatchResult] = None
    error: Optional[str] = None

    def report_row(self) -> Mapping[str, Any]:
        row: dict[str, Any] = dict(image=self.ref.key, status=self.status.value)
        if self.shape:
            row.update(height=self.shape[0], width=self.shape[1])
        if self.result is not None:
            row.update(method=self.result.method, n_instances=self.result.labels.n_instances)
            row.update(self.result.info)
        if self.match is not None:
            row.update(tp=self.match.tp, fp=self.match.fp, fn=self.match.fn)
        if self.status is not Status.SKIPPED:
            row["seconds"] = round(self.seconds, 2)
        if self.error:
            row["error"] = self.error
        return row


# ───────────────────────────────────────────── method comparison ──

@dataclass(frozen=True)
class PairCounts:
    """Same-instance decisions on sampled pairs of ground-truth membrane pixels.

    A pair is positive if both pixels belong to the same GT instance. A method "predicts
    same" if both pixels carry the same non-zero predicted label. Counts are additive, so
    they can be pooled over images.
    """

    tp: int = 0  # same in GT, same in prediction
    fn: int = 0  # same in GT, split by prediction
    fp: int = 0  # different in GT, merged by prediction
    tn: int = 0  # different in GT, kept apart

    def __add__(self, o: "PairCounts") -> "PairCounts":
        return PairCounts(self.tp + o.tp, self.fn + o.fn, self.fp + o.fp, self.tn + o.tn)

    @property
    def auroc(self) -> Optional[float]:
        """ROC AUC of the binary same-instance score (= balanced accuracy for hard labels).
        None if the pairs contain only one class."""
        pos, neg = self.tp + self.fn, self.fp + self.tn
        if pos == 0 or neg == 0:
            return None
        return 0.5 * (self.tp / pos + self.tn / neg)


@dataclass
class MethodOutcome:
    """One method on one image."""

    method: str
    status: Status
    seconds: float = 0.0
    result: Optional[SegmentationResult] = None
    match: Optional[MatchResult] = None
    pairs: Optional[PairCounts] = None
    error: Optional[str] = None

    @property
    def f1(self) -> Optional[float]:
        if self.match is None:
            return None
        m = self.match
        p, r = m.tp / max(m.tp + m.fp, 1), m.tp / max(m.tp + m.fn, 1)
        return 2 * p * r / max(p + r, 1e-9)


@dataclass
class ComparisonItem:
    """All methods on one image."""

    ref: ImageRef
    shape: Optional[tuple[int, int]] = None
    has_gt: bool = False
    n_gt: Optional[int] = None
    outcomes: dict[str, MethodOutcome] = field(default_factory=dict)
    error: Optional[str] = None  # reading the image / ground truth failed


@dataclass
class MethodSummary:
    method: str
    description: str
    n_ok: int
    n_failed: int
    setup_seconds: float
    total_seconds: float
    mean_seconds: float
    total_instances: int
    evaluation: Optional[EvaluationSummary] = None
    pairs: Optional[PairCounts] = None

    @property
    def pair_auroc(self) -> Optional[float]:
        return self.pairs.auroc if self.pairs is not None else None


@dataclass
class ComparisonSummary:
    methods: list[MethodSummary]
    n_images: int
    n_with_gt: int
