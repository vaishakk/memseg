"""Report writers and console progress."""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Optional, Sequence, TextIO

from ..domain.models import ComparisonItem, ComparisonSummary, EvaluationSummary, ItemOutcome, Status
from ..domain.ports import ComparisonListener, ComparisonReportWriter, ProgressListener, ReportWriter


class CsvReportWriter(ReportWriter):
    """summary.csv with one row per image (columns = union of all row keys);
    with an evaluation summary also summary.json."""

    def __init__(self, out_dir: str, name: str = "summary.csv"):
        self.out_dir, self.name = out_dir, name

    def write(self, outcomes: Sequence[ItemOutcome], summary: Optional[EvaluationSummary] = None) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        rows = [dict(o.report_row()) for o in outcomes]
        keys: list[str] = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(self.out_dir, self.name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        if summary is not None:
            with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
                json.dump(summary.as_dict(), f, indent=2)


def _fmt(v) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


class ConsoleProgress(ProgressListener):
    def __init__(self, stream: TextIO = sys.stdout, header: str = ""):
        self.stream, self.header = stream, header

    def _p(self, s: str) -> None:
        print(s, file=self.stream, flush=True)

    def start(self, total: int, description: str) -> None:
        self._p("=" * 64)
        if self.header:
            self._p(self.header)
        self._p(f"{description}: {total} image(s)")
        self._p("=" * 64)

    def item_done(self, index: int, total: int, o: ItemOutcome) -> None:
        head = f"[{index}/{total}] {o.ref.key}:"
        if o.status is Status.SKIPPED:
            self._p(f"{head} output exists, skipped")
        elif o.status is Status.FAILED:
            self._p(f"{head} FAILED  {o.error}")
        else:
            r = o.result
            info = "  ".join(f"{k} {_fmt(v)}" for k, v in r.info.items())
            line = f"{head} {o.shape[1]}x{o.shape[0]}  {info}  instances {r.labels.n_instances}  {o.seconds:.1f}s"
            if o.match is not None:
                m = o.match
                p = m.tp / max(m.tp + m.fp, 1)
                rc = m.tp / max(m.tp + m.fn, 1)
                f1 = 2 * p * rc / max(p + rc, 1e-9)
                iou = sum(m.ious) / len(m.ious) if m.ious else 0.0
                line += f"  gt {m.tp + m.fn}  F1 {f1:.3f}  IoU {iou:.3f}"
            self._p(line)

    def finished(self, outcomes: Sequence[ItemOutcome], summary: Optional[EvaluationSummary]) -> None:
        n = {s: sum(o.status is s for o in outcomes) for s in Status}
        self._p(f"done: {n[Status.OK]} ok, {n[Status.FAILED]} failed, {n[Status.SKIPPED]} skipped")
        if summary is not None:
            self._p("OVERALL: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                              for k, v in summary.as_dict().items()}))


# ───────────────────────────────────────────── method comparison ──

def _r(v, n=4):
    return None if v is None else round(v, n)


def comparison_rows(items: Sequence[ComparisonItem]) -> list[dict]:
    """One row per image with <method>_<metric> columns."""
    rows = []
    for it in items:
        row: dict = dict(image=it.ref.key, has_gt=it.has_gt)
        if it.shape:
            row.update(height=it.shape[0], width=it.shape[1])
        if it.n_gt is not None:
            row["gt_instances"] = it.n_gt
        if it.error:
            row["error"] = it.error
        for name, o in it.outcomes.items():
            row[f"{name}_status"] = o.status.value
            row[f"{name}_seconds"] = round(o.seconds, 3)
            if o.result is not None:
                row[f"{name}_instances"] = o.result.labels.n_instances
            if o.match is not None:
                row[f"{name}_f1"] = _r(o.f1)
                row[f"{name}_tp"], row[f"{name}_fp"], row[f"{name}_fn"] = o.match.tp, o.match.fp, o.match.fn
            if o.pairs is not None:
                row[f"{name}_pair_auroc"] = _r(o.pairs.auroc)
            if o.error:
                row[f"{name}_error"] = o.error
        rows.append(row)
    return rows


def comparison_dict(summary: ComparisonSummary) -> dict:
    out = dict(n_images=summary.n_images, n_with_ground_truth=summary.n_with_gt, methods={})
    for m in summary.methods:
        d = dict(description=m.description, ok=m.n_ok, failed=m.n_failed, setup_seconds=_r(m.setup_seconds, 2),
                 total_seconds=_r(m.total_seconds, 2), mean_seconds_per_image=_r(m.mean_seconds, 3),
                 total_instances=m.total_instances)
        if m.evaluation is not None:
            d.update(f1=_r(m.evaluation.f1), precision=_r(m.evaluation.precision),
                     recall=_r(m.evaluation.recall), mean_matched_iou=_r(m.evaluation.mean_matched_iou),
                     pair_auroc=_r(m.pair_auroc))
            if m.pairs is not None:
                d["pairs"] = dict(tp=m.pairs.tp, fn=m.pairs.fn, fp=m.pairs.fp, tn=m.pairs.tn)
        out["methods"][m.method] = d
    return out


class ComparisonCsvReport(ComparisonReportWriter):
    """comparison.csv (per image) and comparison.json (per method)."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir

    def write(self, items: Sequence[ComparisonItem], summary: ComparisonSummary) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        rows = comparison_rows(items)
        keys: list[str] = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(self.out_dir, "comparison.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(self.out_dir, "comparison.json"), "w") as f:
            json.dump(comparison_dict(summary), f, indent=2)


class ConsoleComparison(ComparisonListener):
    """Per-image lines while running, then a side-by-side results table."""

    def __init__(self, stream: TextIO = sys.stdout, header: str = ""):
        self.stream, self.header = stream, header

    def _p(self, s: str = "") -> None:
        print(s, file=self.stream, flush=True)

    def start(self, total: int, methods: Sequence[str]) -> None:
        self._p("=" * 72)
        if self.header:
            self._p(self.header)
        self._p(f"comparing {' vs '.join(methods)} on {total} image(s)")
        self._p("=" * 72)

    def item_done(self, index: int, total: int, it: ComparisonItem) -> None:
        head = f"[{index}/{total}] {it.ref.key}"
        if it.error:
            self._p(f"{head}: FAILED to read  {it.error}")
            return
        parts = []
        for name, o in it.outcomes.items():
            if o.status is not Status.OK:
                parts.append(f"{name}: FAILED ({o.error})")
                continue
            s = f"{name}: {o.result.labels.n_instances} inst {o.seconds:.2f}s"
            if o.match is not None:
                s += f" F1 {o.f1:.3f}"
                if o.pairs is not None and o.pairs.auroc is not None:
                    s += f" AUROC {o.pairs.auroc:.3f}"
            parts.append(s)
        gt = f"  gt {it.n_gt}" if it.has_gt else ""
        self._p(f"{head}{gt}  |  " + "  |  ".join(parts))

    def finished(self, items: Sequence[ComparisonItem], summary: ComparisonSummary) -> None:
        ms = summary.methods
        w = max(14, *(len(m.method) + 2 for m in ms))
        rows: list[tuple[str, list[str]]] = [
            ("images ok / failed", [f"{m.n_ok} / {m.n_failed}" for m in ms]),
            ("instances (total)", [str(m.total_instances) for m in ms]),
            ("setup time", [f"{m.setup_seconds:.2f} s" for m in ms]),
            ("run time total", [f"{m.total_seconds:.2f} s" for m in ms]),
            ("run time / image", [f"{m.mean_seconds:.2f} s" for m in ms]),
        ]
        has_gt = any(m.evaluation is not None for m in ms)
        if has_gt:
            def f(v):
                return "-" if v is None else f"{v:.4f}"
            rows += [
                ("F1", [f(m.evaluation.f1 if m.evaluation else None) for m in ms]),
                ("precision", [f(m.evaluation.precision if m.evaluation else None) for m in ms]),
                ("recall", [f(m.evaluation.recall if m.evaluation else None) for m in ms]),
                ("mean matched IoU", [f(m.evaluation.mean_matched_iou if m.evaluation else None) for m in ms]),
                ("pair AUROC", [f(m.pair_auroc) for m in ms]),
            ]
        self._p()
        self._p("RESULTS" + (f"  (ground truth for {summary.n_with_gt}/{summary.n_images} images)"
                             if has_gt else "  (no ground truth: timing and counts only)"))
        self._p(f"{'':22s}" + "".join(f"{m.method:>{w}s}" for m in ms))
        for label, vals in rows:
            self._p(f"{label:22s}" + "".join(f"{v:>{w}s}" for v in vals))
        ok = [m for m in ms if m.n_ok and m.mean_seconds > 0]
        if len(ok) >= 2:
            fast, slow = min(ok, key=lambda m: m.mean_seconds), max(ok, key=lambda m: m.mean_seconds)
            self._p(f"\n{fast.method} is {slow.mean_seconds / fast.mean_seconds:.1f}x faster than {slow.method} "
                    f"per image (excluding setup)")
        if has_gt:
            scored = [m for m in ms if m.evaluation is not None]
            best = max(scored, key=lambda m: m.evaluation.f1)
            self._p(f"best F1: {best.method} ({best.evaluation.f1:.4f})")
            self._p("pair AUROC: same-instance decision on sampled nearby pairs of ground-truth membrane "
                    "pixels; methods give hard labels, so it equals balanced accuracy on those pairs")
