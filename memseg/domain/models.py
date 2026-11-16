"""Domain models. Plain data, no I/O, no framework dependencies (numpy only)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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
    gt_path: str | None = None


@dataclass
class MembraneMask:
    """Binary membrane mask: True on membrane pixels."""

    data: np.ndarray
    ref: ImageRef | None = None

    def __post_init__(self):
        a = np.asarray(self.data)
        if a.ndim != 2:
            raise ValueError(f"membrane mask must be 2D, got shape {a.shape}")
        self.data = a.astype(bool, copy=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

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
    def consecutive(cls, data: np.ndarray) -> InstanceLabels:
        """Relabel any integer map to consecutive ids 1..N (negative/zero -> background)."""
        # plain np.ndarray: numpy 2.5 stubs lack a scalar max() overload for int64-typed arrays
        a: np.ndarray = np.asarray(data).astype(np.int64)
        a[a < 0] = 0
        ids = np.unique(a)
        ids = ids[ids > 0]
        lut = np.zeros(int(a.max()) + 1 if a.size else 1, np.int32)
        lut[ids] = np.arange(1, len(ids) + 1)
        return cls(lut[a])


@dataclass
class Overlay:
    """Optional visual diagnostics a method can attach (drawn on previews)."""

    skeleton: np.ndarray | None = None  # bool (H, W)
    junctions: np.ndarray | None = None  # bool (H, W)
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
    overlay: Overlay | None = None


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
    recall_touching: float | None = None

    def as_dict(self) -> dict[str, float]:
        d = dict(
            precision=self.precision,
            recall=self.recall,
            f1=self.f1,
            mean_matched_iou=self.mean_matched_iou,
            n_images=self.n_images,
        )
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
    shape: tuple[int, int] | None = None
    result: SegmentationResult | None = None
    match: MatchResult | None = None
    error: str | None = None

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

    def __add__(self, o: PairCounts) -> PairCounts:
        return PairCounts(self.tp + o.tp, self.fn + o.fn, self.fp + o.fp, self.tn + o.tn)

    @property
    def auroc(self) -> float | None:
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
    result: SegmentationResult | None = None
    match: MatchResult | None = None
    pairs: PairCounts | None = None
    error: str | None = None

    @property
    def f1(self) -> float | None:
        if self.match is None:
            return None
        m = self.match
        p, r = m.tp / max(m.tp + m.fp, 1), m.tp / max(m.tp + m.fn, 1)
        return 2 * p * r / max(p + r, 1e-9)


@dataclass
class ComparisonItem:
    """All methods on one image."""

    ref: ImageRef
    shape: tuple[int, int] | None = None
    has_gt: bool = False
    n_gt: int | None = None
    outcomes: dict[str, MethodOutcome] = field(default_factory=dict)
    error: str | None = None  # reading the image / ground truth failed


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
    evaluation: EvaluationSummary | None = None
    pairs: PairCounts | None = None

    @property
    def pair_auroc(self) -> float | None:
        return self.pairs.auroc if self.pairs is not None else None


@dataclass
class ComparisonSummary:
    methods: list[MethodSummary]
    n_images: int
    n_with_gt: int


# ───────────────────────────────────────────── semantic segmentation ──


@dataclass
class Micrograph:
    """A raw 2-D image (e.g. a cryo-EM micrograph) to be segmented into membrane / background.

    pixel_size:  Å per pixel if known (e.g. from an MRC header), else None
    """

    data: np.ndarray
    ref: ImageRef | None = None
    pixel_size: float | None = None

    def __post_init__(self):
        a = np.squeeze(np.asarray(self.data))
        if a.ndim != 2:
            raise ValueError(f"micrograph must be 2D, got shape {a.shape}")
        self.data = a.astype(np.float32, copy=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape


@dataclass
class SemanticResult:
    """Output of one semantic method on one micrograph.

    mask:         the membrane mask (the input for an instance method)
    probability:  optional per-pixel membrane probability / score in [0, 1], same shape
    info:         small scalar diagnostics (written to reports)
    artifacts:    optional large intermediate arrays a sink may save
    """

    mask: MembraneMask
    method: str
    probability: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        if self.probability is not None and np.shape(self.probability) != self.mask.shape:
            raise ValueError(f"probability shape {np.shape(self.probability)} != mask shape {self.mask.shape}")


@dataclass(frozen=True)
class PixelScores:
    """Pixel-level confusion counts of a predicted mask against a ground-truth mask.
    Additive, so they can be pooled over images."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __add__(self, o: PixelScores) -> PixelScores:
        return PixelScores(self.tp + o.tp, self.fp + o.fp, self.fn + o.fn, self.tn + o.tn)

    @property
    def iou(self) -> float:
        return self.tp / max(self.tp + self.fp + self.fn, 1)

    @property
    def dice(self) -> float:
        return 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1)

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)


@dataclass(frozen=True)
class SemanticSummary:
    """Pixel scores pooled over all images, plus the per-image mean IoU."""

    pooled: PixelScores
    mean_iou: float
    n_images: int

    def as_dict(self) -> dict[str, float]:
        p = self.pooled
        return dict(
            iou=p.iou,
            dice=p.dice,
            precision=p.precision,
            recall=p.recall,
            mean_image_iou=self.mean_iou,
            n_images=self.n_images,
        )


@dataclass
class MicrographOutcome:
    """What happened to one micrograph: semantic result, optional instances and scores."""

    ref: ImageRef
    status: Status
    seconds: float = 0.0
    shape: tuple[int, int] | None = None
    semantic: SemanticResult | None = None
    instances: SegmentationResult | None = None
    scores: PixelScores | None = None
    error: str | None = None

    def report_row(self) -> Mapping[str, Any]:
        row: dict[str, Any] = dict(image=self.ref.key, status=self.status.value)
        if self.shape:
            row.update(height=self.shape[0], width=self.shape[1])
        if self.semantic is not None:
            s = self.semantic
            row.update(semantic_method=s.method, membrane_fraction=round(float(s.mask.data.mean()), 5))
            row.update({f"semantic_{k}": v for k, v in s.info.items()})
        if self.instances is not None:
            r = self.instances
            row.update(instance_method=r.method, n_instances=r.labels.n_instances)
            row.update({f"instance_{k}": v for k, v in r.info.items()})
        if self.scores is not None:
            row.update(iou=round(self.scores.iou, 4), dice=round(self.scores.dice, 4))
        if self.status is not Status.SKIPPED:
            row["seconds"] = round(self.seconds, 2)
        if self.error:
            row["error"] = self.error
        return row
