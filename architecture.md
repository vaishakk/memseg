# memseg architecture

`memseg` turns binary membrane masks into instance label maps (one label per membrane,
one label per layer of a multilamellar vesicle).

Every segmentation method sits behind one interface, `InstanceSegmenter`. Methods are plugins:
the built-in reference method `components` ships with the package, and other packages add
methods through the `memseg.segmenters` entry-point group. Everything else (reading files,
choosing images, writing outputs, scoring against ground truth, the CLI) is shared and
does not know which method runs.

## Layers and the dependency rule

```
                ┌──────────────────────────────────────────────┐
                │ cli.py  (composition root)                    │
                │ picks method plugin + adapters + use case     │
                └───────┬───────────────┬───────────────┬───────┘
                        │               │               │
          ┌─────────────▼───┐   ┌───────▼───────┐   ┌───▼────────────────────┐
          │ io/  (adapters) │   │ application/  │   │ segmenters/ (adapters)  │
          │ files, previews │   │ use cases     │   │ registry + plugins      │
          └─────────┬───────┘   └───────┬───────┘   └───┬────────────────────┘
                    │ implements        │ uses          │ implements
                    └──────────►┌───────▼───────┐◄──────┘
                                │ domain/       │
                                │ models, ports,│
                                │ metrics       │
                                └───────────────┘
```

* Arrows point inwards only. `domain` imports nothing from the package (numpy/scipy only).
  `application` imports only `domain`. `io` and `segmenters` import `domain` (and `segmenters.base`),
  never each other and never `application`. Only `cli.py` imports concrete classes from all layers.
* **The segmentation method is an inverted dependency**: use cases depend on the abstract
  `domain.ports.InstanceSegmenter`; concrete methods implement it and are injected by `cli.py`.
  A use case cannot tell (and must not care) whether a network or a rule-based tracer runs.
* Heavy dependencies stay at the edge: a method imports e.g. torch inside its own package, and
  plugins are loaded lazily through the registry (`get_plugin`), so installing or listing a
  method never imports another method's dependencies.

## Package layout

```
memseg/
├── __main__.py                 python -m memseg ...
├── cli.py                      composition root (argument parsing, wiring)
├── domain/                     CORE — no I/O
│   ├── models.py               ImageRef, MembraneMask, InstanceLabels, SegmentationResult, Overlay,
│   │                           MatchResult, EvaluationSummary, ItemOutcome, Status,
│   │                           PairCounts, MethodOutcome, ComparisonItem, MethodSummary, ComparisonSummary
│   ├── ports.py                InstanceSegmenter, ImageSource, MaskReader, GroundTruthReader,
│   │                           ResultSink (+ CompositeSink, NullSink), ReportWriter, ProgressListener,
│   │                           ComparisonSink, ComparisonReportWriter, ComparisonListener
│   └── metrics.py              match_instances (Hungarian on IoU), summarize,
│                               sample_pixel_pairs + pair_counts (pair AUROC)
├── application/                CORE — use cases
│   ├── use_cases.py            SegmentImages, EvaluateSegmenter
│   └── compare.py              CompareSegmenters, PairSampling
├── io/                         adapters for the file system
│   ├── images.py               ImageFileMaskReader, ImageFileGroundTruthReader, write_labels
│   ├── sources.py              FolderImageSource, LabelledDatasetSource (masks/ + labels/)
│   ├── sinks.py                LabelTiffSink, PreviewSink, ArtifactSink
│   ├── previews.py             colour rendering (+ optional skeleton/junction overlay, mono masks)
│   ├── side_by_side.py         SideBySideSink: captioned panels mask | GT | method A | method B ...
│   └── reports.py              CsvReportWriter (summary.csv/.json), ConsoleProgress,
│                               ComparisonCsvReport (comparison.csv/.json), ConsoleComparison (results table)
└── segmenters/                 adapters: segmentation methods
    ├── __init__.py             registry: REGISTRY (built-ins, run-time registration) +
    │                           entry points in group "memseg.segmenters"; lazy import
    ├── base.py                 SegmenterPlugin (CLI options + factory, summary), --config/--set parsing
    └── components.py           reference method: one instance per connected piece
tests/test_core.py              use cases against fakes (no files, no network)
tests/test_plugins.py           registry, entry-point discovery, reference method
```

## Core models (`domain/models.py`)

| model | meaning |
|---|---|
| `ImageRef(path, key, gt_path)` | one input; `key` is the relative name outputs mirror (`sub/mask_01.tif` → `OUT/sub/mask_01.tif`) |
| `MembraneMask(data, ref)` | 2-D bool mask, validated on construction |
| `InstanceLabels(data)` | 2-D int32 map, 0 = background; `consecutive()` relabels to 1..N |
| `SegmentationResult(labels, method, info, artifacts, overlay)` | method output. `info`: small scalars for logs/reports (e.g. `scale`, `junctions`); `artifacts`: large optional arrays (e.g. network outputs) for `--save_raw`; `overlay`: optional preview hints (skeleton, junctions) |
| `MatchResult`, `EvaluationSummary` | per-image and overall precision / recall / F1 / matched IoU |
| `PairCounts(tp, fn, fp, tn)` | same-instance decisions on sampled pixel pairs; additive over images; `.auroc` |
| `MethodOutcome`, `ComparisonItem` | one method on one image (time, result, match, pairs) / all methods on one image |
| `MethodSummary`, `ComparisonSummary` | per-method totals: setup and run time, instances, F1 etc., pooled pair AUROC |
| `ItemOutcome(ref, status, …)` | what happened to one image (ok / skipped / failed), turns itself into a report row |

## Ports (`domain/ports.py`)

| port | implemented by | purpose |
|---|---|---|
| `InstanceSegmenter.segment(mask) -> SegmentationResult` | `ConnectedComponentsSegmenter`, plugin packages | the method; must do no file I/O |
| `ImageSource` (iterable of `ImageRef`) | `FolderImageSource`, `LabelledDatasetSource` | which images; also handles `--sample/--seed/--limit` |
| `MaskReader`, `GroundTruthReader` | `ImageFile*Reader` | load masks / GT labels |
| `ResultSink.exists/save` | `LabelTiffSink`, `PreviewSink`, `ArtifactSink`, `CompositeSink`, `NullSink` | persist results; the first sink of a `CompositeSink` decides "already done" |
| `ReportWriter` | `CsvReportWriter` | `summary.csv` (+ `summary.json` when evaluating) |
| `ProgressListener` | `ConsoleProgress` | log lines; a GUI or plugin host can supply its own |
| `ComparisonSink` | `SideBySideSink` | one side-by-side preview per compared image |
| `ComparisonReportWriter` | `ComparisonCsvReport` | `comparison.csv` (per image) + `comparison.json` (per method) |
| `ComparisonListener` | `ConsoleComparison` | per-image lines and the final results table |

## Use cases (`application/use_cases.py`)

* **`SegmentImages`**(segmenter, source, reader, sink, report, progress, skip_existing):
  for each image: skip if the sink already has it → read → segment → save. A failure on one image
  is recorded as `FAILED` and the run continues.
* **`EvaluateSegmenter`**(segmenter, source, reader, gt_reader, sink, report, progress, iou_threshold):
  read mask + GT → segment → optionally save → `match_instances` → `summarize`.

* **`CompareSegmenters`**(segmenters: name → method, source, reader, gt_reader, method_sinks,
  comparison_sink, report, progress, setup_seconds, iou_threshold, pairs, warmup):
  per image: read mask and GT (if `ref.gt_path`), sample pixel pairs once, run every method
  (timed with `perf_counter`, segmentation only), score F1 and pair counts when GT exists, save each
  method's labels, then hand all outcomes to the comparison sink. `warmup` runs every method once on
  the first image, untimed, so GPU/cuDNN start-up does not count. Setup time (e.g. loading the network)
  is measured by the composition root and reported separately.

None of them imports files, a network library or a concrete method; `tests/test_core.py` runs both with
in-memory fakes.

## Command line

```bash
python -m memseg methods
python -m memseg segment  masks/ -o out --method components --preview
python -m memseg segment  masks/ -o out --method components --preview_dir prev --set min_size=200
python -m memseg evaluate data/val --method components --limit 20 -o eval_out
python -m memseg compare  data/val -o cmp --methods components my_method --warmup
python -m memseg compare  masks/ --gt_dir labels/ -o cmp --methods components my_method --set components.min_size=200
python -m memseg segment --method components --help   # shows the method's own options
```

### `compare`
Runs two or more methods (`--methods`, default: all installed methods) on the same masks.

* Inputs: plain mask files/folders (optionally with `--gt_dir`, matched by relative name, `mask_X` →
  `label_X` accepted), or labelled dataset folders (`masks/` + `labels/`, ground truth used
  automatically). Mixing the two kinds in one call is rejected.
* Method options: all selected methods' flags are accepted (e.g. a checkpoint path for a network method); per-method
  parameters use `--set METHOD.KEY=VALUE` and `--config METHOD=FILE`.
* Output folder: `side_by_side/<key>.png` (input mask | ground truth | one panel per method, each
  captioned with instances, run time and, with GT, F1 and pair AUROC), `<method>/<key>.tif` label maps
  (`--no_labels` to skip), `comparison.csv` (one row per image, `<method>_<metric>` columns) and
  `comparison.json` (per-method summary incl. pair counts).
* Console: one line per image, then a table with images ok/failed, instances, setup time, total and
  per-image run time, and with GT: F1, precision, recall, mean matched IoU, pair AUROC; plus the speed
  ratio and the best F1.
* Panel colours are per method: the same instance can have different colours in different panels.

**Pair AUROC.** Neither method outputs confidences, only labels, so AUROC is computed on the decision
both methods make: "are these two membrane pixels the same instance?". `sample_pixel_pairs` draws
`--pairs` (default 20 000) pairs of ground-truth membrane pixels at most `--pair_radius` (100 px) apart,
once per image and identical for every method (seeded by `--seed`). A pair is positive when both pixels
have the same GT label; a method says "same" when both carry the same non-zero label. With binary scores
the ROC curve has a single point, so AUROC = (TPR + TNR) / 2, i.e. balanced accuracy over pairs. Nearby
pairs are used because far-apart pairs are trivially "different" for every method; the metric is most
sensitive to splits (a ring broken in two) and merges (touching rings or stacked layers joined). If a
method later provides a continuous same-instance score, `pair_counts` can be replaced by a true ROC
computation without changing the use case.

Shared options: `--method`, `--config JSON`, `--set KEY=VALUE` (repeatable), `--preview_dir`,
`--limit`, `--sample`, `--seed`; `segment` adds `-o`, `--preview`, `--save_raw`, `--overwrite`,
`--ext`, `--no_recursive`; `evaluate` adds `-o`, `--iou`.

Outputs: `<key>.tif` (uint16, uint32 if > 65 535 instances), `<key>_preview.png`, `<key>_raw.npz`,
`summary.csv` (one row per image: status, size, method info, n_instances, seconds, tp/fp/fn),
`summary.json` (evaluation).

## Extending

**A new segmentation method**
1. Write a class implementing `InstanceSegmenter.segment(mask) -> SegmentationResult` (no file I/O;
   put diagnostics in `info`, big arrays in `artifacts`, drawing hints in `overlay`).
2. Add a `SegmenterPlugin` subclass: `summary`, `add_arguments(group)` for CLI flags,
   `build(args, overrides)` to construct the segmenter.
3. Register it, preferably from its own package via an entry point:
   ```toml
   [project.entry-points."memseg.segmenters"]
   mymethod = "mypackage.segmenter:MyPlugin"
   ```
   Alternatively, add it to `segmenters.REGISTRY` at run time, or (for built-ins) in
   `segmenters/__init__.py`. A built-in name wins over an entry point with the same name.
   Nothing in `domain`, `application` or `io` changes. The method's package can stay private;
   the framework only needs it installed.

**A new input/output** (e.g. MRC files, a CryoSPARC job, a database): implement the matching port
(`ImageSource`, `MaskReader`, `ResultSink`, `ReportWriter`, `ProgressListener`) under `io/` or in the
host application, and wire it in a composition root (`cli.py` or the host's own). Use cases and methods
stay untouched; e.g. a CryoSPARC plugin can call `SegmentImages(...)` with its own source and sink.

**Using a method from Python**
```python
from memseg.segmenters.components import ConnectedComponentsSegmenter
from memseg.domain import MembraneMask
result = ConnectedComponentsSegmenter(min_size=50).segment(MembraneMask(mask_array))
labels = result.labels.data
```

## Design notes

* Previews use one renderer for every method (golden-ratio palette, brightness max-pooling when
  downsizing so thin membranes stay visible).
* Scoring (`domain/metrics.py`) is part of the core, so every method is scored the same way.
* Plugin discovery never imports a method until it is used; a plugin that fails to import shows up
  in `memseg methods` as "failed to load" instead of breaking the CLI.
