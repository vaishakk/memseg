"""Command line interface = composition root.

This is the only module that knows concrete classes from every layer: it picks the
segmentation method (plugin), the file adapters and the use case, and wires them.

    python -m memseg methods
    python -m memseg segment  INPUT... -o OUT --method METHOD [--preview] [--set KEY=VALUE]
    python -m memseg evaluate DATASET... --method METHOD --limit 20 [-o OUT]
    python -m memseg compare  INPUT... -o OUT --methods METHOD_A METHOD_B

`python -m memseg segment --method METHOD --help` lists the method's own options.
Methods are plugins: see memseg.segmenters.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

from . import segmenters
from .application import CompareSegmenters, EvaluateSegmenter, PairSampling, SegmentImages
from .domain import CompositeSink, NullSink, Status
from .io import (IMAGE_EXTS, ArtifactSink, ComparisonCsvReport, ConsoleComparison, ConsoleProgress,
                 CsvReportWriter, FolderImageSource, ImageFileGroundTruthReader, ImageFileMaskReader,
                 LabelledDatasetSource, LabelTiffSink, PreviewSink, SideBySideSink)
from .io.sources import is_labelled_dataset
from .segmenters.base import load_overrides

DEFAULT_METHOD = "components"


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--method", "-m", default=DEFAULT_METHOD, choices=segmenters.available(),
                   help="segmentation method")
    p.add_argument("--config", default=None, help="JSON file with method parameters")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", default=[],
                   help="override one method parameter (repeatable), e.g. --set min_size=100")
    p.add_argument("--preview_dir", default=None, help="write colour PNG previews here")
    p.add_argument("--limit", type=int, default=None, help="process at most N images")
    p.add_argument("--sample", type=int, default=None, help="process N randomly chosen images")
    p.add_argument("--seed", type=int, default=0, help="random seed for --sample")


def build_parser(method: Optional[str] = None, compare_methods: Sequence[str] = ()) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="memseg", description="Membrane instance segmentation.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("methods", help="list available segmentation methods")

    s = sub.add_parser("segment", help="segment masks and write instance label maps",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    s.add_argument("inputs", nargs="+", help="mask files or folders (searched recursively)")
    s.add_argument("--output_dir", "-o", default=None, help="write label maps (+ summary.csv) here")
    _common(s)
    s.add_argument("--preview", action="store_true", help="also write <name>_preview.png into --output_dir")
    s.add_argument("--save_raw", action="store_true", help="also write method intermediates (<name>_raw.npz)")
    s.add_argument("--overwrite", action="store_true", help="recompute images whose output exists")
    s.add_argument("--ext", default=",".join(IMAGE_EXTS), help="comma-separated input extensions")
    s.add_argument("--no_recursive", action="store_true", help="do not descend into sub-folders")

    e = sub.add_parser("evaluate", help="segment labelled datasets (masks/ + labels/) and report F1 / IoU",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    e.add_argument("datasets", nargs="+", help="dataset folders containing masks/ and labels/")
    e.add_argument("--output_dir", "-o", default=None,
                   help="write predicted label maps, summary.csv and summary.json here")
    e.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a match")
    _common(e)

    c = sub.add_parser("compare", help="run several methods on the same masks and compare results and run time",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                       description="Runs every method on every mask, times it, scores it when ground truth "
                                   "is available (F1 and pair AUROC), and writes side-by-side previews.")
    c.add_argument("inputs", nargs="+",
                   help="mask files/folders, or labelled dataset folders (masks/ + labels/: ground truth is used)")
    c.add_argument("--output_dir", "-o", required=True,
                   help="output folder: side_by_side/, <method>/ label maps, comparison.csv/.json")
    c.add_argument("--methods", nargs="+", default=segmenters.available(), choices=segmenters.available(),
                   help="methods to compare (at least two; default: all installed)")
    c.add_argument("--gt_dir", default=None,
                   help="ground-truth label maps for plain mask folders (same relative names; mask_X -> label_X ok)")
    c.add_argument("--config", action="append", default=[], metavar="METHOD=FILE",
                   help="JSON parameters for one method (repeatable)")
    c.add_argument("--set", action="append", default=[], metavar="METHOD.KEY=VALUE",
                   help="override one parameter of one method (repeatable), e.g. --set components.min_size=100")
    c.add_argument("--iou", type=float, default=0.5, help="IoU threshold for an instance match (F1)")
    c.add_argument("--pairs", type=int, default=20000, help="pixel pairs per image for the pair AUROC")
    c.add_argument("--pair_radius", type=float, default=100.0, help="max distance (px) between paired pixels")
    c.add_argument("--panel_size", type=int, default=900, help="max side (px) of each preview panel")
    c.add_argument("--no_labels", action="store_true", help="do not write per-method label maps")
    c.add_argument("--warmup", action="store_true",
                   help="run each method once on the first image before timing (excludes GPU start-up)")
    c.add_argument("--ext", default=",".join(IMAGE_EXTS), help="comma-separated input extensions")
    c.add_argument("--no_recursive", action="store_true", help="do not descend into sub-folders")
    c.add_argument("--limit", type=int, default=None, help="process at most N images")
    c.add_argument("--sample", type=int, default=None, help="process N randomly chosen images")
    c.add_argument("--seed", type=int, default=0, help="random seed for --sample and pair sampling")
    for m in dict.fromkeys(compare_methods):
        group = c.add_argument_group(f"{m} method options")
        try:
            segmenters.get_plugin(m).add_arguments(group)
        except argparse.ArgumentError:
            pass  # an option with the same name is already declared by another method: it is shared

    if method:  # method-specific options, shown in --help once the method is known
        plugin = segmenters.get_plugin(method)
        for p in (s, e):
            plugin.add_arguments(p.add_argument_group(f"{method} method options"))
    return ap


def _peek_method(argv: Sequence[str]) -> str:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--method", "-m", default=DEFAULT_METHOD)
    known, _ = pre.parse_known_args(argv)
    return known.method


def _peek_methods(argv: Sequence[str]) -> list[str]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--methods", nargs="+", default=segmenters.available())
    known, _ = pre.parse_known_args(argv)
    return known.methods


def _per_method_overrides(methods: Sequence[str], configs: Sequence[str], sets: Sequence[str]) -> dict:
    """--config METHOD=FILE and --set METHOD.KEY=VALUE -> {method: overrides}."""
    out: dict[str, dict] = {m: {} for m in methods}
    for c in configs:
        if "=" not in c or c.split("=", 1)[0] not in out:
            raise SystemExit(f"--config expects METHOD=FILE with METHOD in {list(out)}, got {c!r}")
        m, path = c.split("=", 1)
        out[m].update(load_overrides(path, None))
    for s_ in sets:
        head = s_.split("=", 1)[0]
        if "." not in head or head.split(".", 1)[0] not in out:
            raise SystemExit(f"--set expects METHOD.KEY=VALUE with METHOD in {list(out)}, got {s_!r}")
        m, rest = s_.split(".", 1)
        out[m].update(load_overrides(None, [rest]))
    return out


def run_compare(args) -> int:
    methods = list(dict.fromkeys(args.methods))
    if len(methods) < 2:
        print("--methods needs at least two different methods", file=sys.stderr)
        return 2
    out_dir = os.path.abspath(args.output_dir)
    overrides = _per_method_overrides(methods, args.config, args.set)

    built, setup = {}, {}
    for m in methods:
        t0 = time.perf_counter()
        built[m] = segmenters.get_plugin(m).build(args, overrides[m])
        setup[m] = time.perf_counter() - t0

    datasets = [p for p in args.inputs if os.path.isdir(p) and is_labelled_dataset(p)]
    if datasets and len(datasets) != len(args.inputs):
        print("mix of labelled dataset folders and plain mask inputs; run them separately", file=sys.stderr)
        return 2
    if datasets:
        source = LabelledDatasetSource(datasets, sample=args.sample, seed=args.seed, limit=args.limit)
    else:
        exts = [x.strip() for x in args.ext.split(",") if x.strip()]
        source = FolderImageSource(args.inputs, exts, recursive=not args.no_recursive, exclude_dirs=[out_dir],
                                   sample=args.sample, seed=args.seed, limit=args.limit, gt_dir=args.gt_dir)

    header = "\n".join(f"{m:12s}{built[m].describe()}  (setup {setup[m]:.1f} s)" for m in methods)
    use_case = CompareSegmenters(
        segmenters=built, source=source, reader=ImageFileMaskReader(), gt_reader=ImageFileGroundTruthReader(),
        method_sinks={} if args.no_labels else {m: LabelTiffSink(os.path.join(out_dir, m)) for m in methods},
        comparison_sink=SideBySideSink(os.path.join(out_dir, "side_by_side"), panel_size=args.panel_size),
        report=ComparisonCsvReport(out_dir),
        progress=ConsoleComparison(header=header + f"\noutput      {out_dir}"),
        setup_seconds=setup, iou_threshold=args.iou,
        pairs=PairSampling(n_pairs=args.pairs, radius=args.pair_radius, seed=args.seed), warmup=args.warmup)
    items, _ = use_case.run()
    if not items:
        print("no images found", file=sys.stderr)
        return 1
    return 0 if any(o.status is Status.OK for it in items for o in it.outcomes.values()) else 1


def _sinks(out_dir: Optional[str], preview: bool, preview_dir: Optional[str], save_raw: bool):
    sinks = []
    if out_dir:
        sinks.append(LabelTiffSink(out_dir))
        if preview:
            sinks.append(PreviewSink(out_dir, "_preview.png"))
        if save_raw:
            sinks.append(ArtifactSink(out_dir))
    if preview_dir:
        sinks.append(PreviewSink(preview_dir, ".png"))
    return CompositeSink(sinks) if sinks else NullSink()


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "methods":
        for name in segmenters.available():
            print(f"{name:12s} {segmenters.summary(name)}")
        return 0
    if argv and argv[0] == "compare":
        methods = _peek_methods(argv)
        bad = [m for m in methods if m not in segmenters.available()]
        if bad:
            print(f"unknown method(s) {bad}; available: {', '.join(segmenters.available())}", file=sys.stderr)
            return 2
        return run_compare(build_parser(compare_methods=methods).parse_args(argv))
    method = _peek_method(argv)
    if method not in segmenters.available():
        print(f"unknown method {method!r}; available: {', '.join(segmenters.available())}", file=sys.stderr)
        return 2
    args = build_parser(method).parse_args(argv)

    plugin = segmenters.get_plugin(args.method)
    segmenter = plugin.build(args, load_overrides(args.config, args.set))
    reader = ImageFileMaskReader()
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else None
    report = CsvReportWriter(out_dir) if out_dir else None
    progress = ConsoleProgress(header=f"method      {segmenter.describe()}\noutput      {out_dir or '(none)'}")

    if args.command == "segment":
        for inp in args.inputs:
            if out_dir and os.path.isdir(inp) and os.path.abspath(inp) == out_dir:
                print("--output_dir must differ from the input folder", file=sys.stderr)
                return 2
        exts = [x.strip() for x in args.ext.split(",") if x.strip()]
        source = FolderImageSource(args.inputs, exts, recursive=not args.no_recursive,
                                   exclude_dirs=[d for d in (out_dir, args.preview_dir) if d],
                                   sample=args.sample, seed=args.seed, limit=args.limit)
        outcomes = SegmentImages(segmenter, source, reader,
                                 sink=_sinks(out_dir, args.preview, args.preview_dir, args.save_raw),
                                 report=report, progress=progress, skip_existing=not args.overwrite).run()
    else:
        source = LabelledDatasetSource(args.datasets, sample=args.sample, seed=args.seed, limit=args.limit)
        outcomes, _ = EvaluateSegmenter(segmenter, source, reader, ImageFileGroundTruthReader(),
                                        sink=_sinks(out_dir, False, args.preview_dir, False),
                                        report=report, progress=progress, iou_threshold=args.iou).run()

    if not outcomes:
        print("no images found", file=sys.stderr)
        return 1
    ok = sum(o.status is Status.OK for o in outcomes)
    skipped = sum(o.status is Status.SKIPPED for o in outcomes)
    return 0 if ok or skipped == len(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
