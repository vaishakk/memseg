"""Instance matching metrics (Hungarian matching on IoU), independent of any method."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .models import (EvaluationSummary, InstanceLabels, MatchResult, MembraneMask, PairCounts, PixelScores,
                     SemanticSummary)


def match_instances(gt: InstanceLabels, pred: InstanceLabels, iou_threshold: float = 0.5) -> MatchResult:
    """One-to-one matching maximising IoU; a match counts if IoU >= iou_threshold."""
    g, p = gt.data, pred.data
    if g.shape != p.shape:
        raise ValueError(f"shape mismatch: gt {g.shape} vs prediction {p.shape}")
    g_ids = np.unique(g)
    g_ids = g_ids[g_ids > 0]
    p_ids = np.unique(p)
    p_ids = p_ids[p_ids > 0]
    if len(g_ids) == 0 or len(p_ids) == 0:
        return MatchResult(tp=0, fp=len(p_ids), fn=len(g_ids), ious=(), matched_gt=frozenset())
    fg = (g > 0) | (p > 0)
    gi = np.searchsorted(g_ids, g[fg])
    pi = np.searchsorted(p_ids, p[fg])
    gv, pv = g[fg] > 0, p[fg] > 0
    inter = np.zeros((len(g_ids), len(p_ids)), np.int64)
    both = gv & pv
    np.add.at(inter, (gi[both], pi[both]), 1)
    g_area = np.bincount(gi[gv], minlength=len(g_ids))
    p_area = np.bincount(pi[pv], minlength=len(p_ids))
    iou = inter / (g_area[:, None] + p_area[None, :] - inter + 1e-9)
    r, c = linear_sum_assignment(-iou)
    ok = iou[r, c] >= iou_threshold
    tp = int(ok.sum())
    return MatchResult(tp=tp, fp=len(p_ids) - tp, fn=len(g_ids) - tp,
                       ious=tuple(float(v) for v in iou[r, c][ok]),
                       matched_gt=frozenset(int(g_ids[i]) for i in r[ok]))


def summarize(results: Sequence[MatchResult],
              touching: Optional[Sequence[set[int]]] = None) -> EvaluationSummary:
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    ious = [v for r in results for v in r.ious]
    rt = None
    if touching is not None:
        total = sum(len(s) for s in touching)
        hit = sum(len(s & r.matched_gt) for s, r in zip(touching, results))
        rt = float(hit / max(total, 1))
    return EvaluationSummary(precision=float(precision), recall=float(recall), f1=float(f1),
                             mean_matched_iou=float(np.mean(ious)) if ious else 0.0,
                             n_images=len(results), recall_touching=rt)


def sample_pixel_pairs(gt: InstanceLabels, n_pairs: int = 20000, radius: float = 100.0,
                       seed: int = 0, max_rounds: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Random pairs of ground-truth membrane pixels at most `radius` px apart.

    Nearby pairs are where methods differ (touching rings, stacked layers, gaps); pairs on
    far-apart vesicles would be trivially "different" for every method. Returns two (k, 2)
    arrays of (y, x) coordinates. The same pairs must be used for every method compared.
    """
    g = gt.data
    ys, xs = np.nonzero(g)
    if len(ys) < 2 or n_pairs <= 0:
        e = np.zeros((0, 2), int)
        return e, e
    rng = np.random.default_rng(seed)
    H, W = g.shape
    got_a, got_b, n = [], [], 0
    for _ in range(max_rounds):  # thin membranes: most random partners land on background
        a = rng.integers(0, len(ys), 4 * n_pairs)
        ang = rng.uniform(0, 2 * np.pi, len(a))
        rad = radius * np.sqrt(rng.uniform(0, 1, len(a)))
        by = np.round(ys[a] + rad * np.sin(ang)).astype(int)
        bx = np.round(xs[a] + rad * np.cos(ang)).astype(int)
        ok = (by >= 0) & (by < H) & (bx >= 0) & (bx < W)
        ok[ok] = g[by[ok], bx[ok]] > 0
        ok &= (by != ys[a]) | (bx != xs[a])
        idx = np.flatnonzero(ok)[: n_pairs - n]
        got_a.append(np.c_[ys[a[idx]], xs[a[idx]]])
        got_b.append(np.c_[by[idx], bx[idx]])
        n += len(idx)
        if n >= n_pairs:
            break
    return np.concatenate(got_a), np.concatenate(got_b)


def pair_counts(gt: InstanceLabels, pred: InstanceLabels, pa: np.ndarray, pb: np.ndarray) -> PairCounts:
    """Score the same-instance decision of a prediction on given pixel pairs."""
    if len(pa) == 0:
        return PairCounts()
    g, p = gt.data, pred.data
    same_gt = g[pa[:, 0], pa[:, 1]] == g[pb[:, 0], pb[:, 1]]
    la, lb = p[pa[:, 0], pa[:, 1]], p[pb[:, 0], pb[:, 1]]
    same_pred = (la == lb) & (la > 0)
    return PairCounts(tp=int((same_gt & same_pred).sum()), fn=int((same_gt & ~same_pred).sum()),
                      fp=int((~same_gt & same_pred).sum()), tn=int((~same_gt & ~same_pred).sum()))


# ───────────────────────────────────────────── semantic segmentation ──

def pixel_scores(gt: MembraneMask, pred: MembraneMask) -> PixelScores:
    """Pixel confusion counts of a predicted membrane mask against a ground-truth mask."""
    g, p = gt.data, pred.data
    if g.shape != p.shape:
        raise ValueError(f"shape mismatch: gt {g.shape} vs prediction {p.shape}")
    tp = int(np.count_nonzero(g & p))
    fp = int(np.count_nonzero(p)) - tp
    fn = int(np.count_nonzero(g)) - tp
    return PixelScores(tp=tp, fp=fp, fn=fn, tn=int(g.size) - tp - fp - fn)


def summarize_semantic(scores: Sequence[PixelScores]) -> SemanticSummary:
    pooled = sum(scores, PixelScores())
    mean = float(np.mean([s.iou for s in scores])) if scores else 0.0
    return SemanticSummary(pooled=pooled, mean_iou=mean, n_images=len(scores))
