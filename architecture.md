# memseg architecture

`memseg` segments membranes in two stages:

* **semantic**: micrograph → binary membrane mask;
* **instance**: mask → instance label map, with one label per membrane, or one per layer of a
  multilamellar vesicle.

Each stage has one interface: `SemanticSegmenter` and `InstanceSegmenter`. The stages connect
only through a `MembraneMask`, so any semantic method can feed any instance method.

Methods are plugins:

* built-in reference methods ship with the package: `threshold` (semantic) and `components` (instance);
* other packages add methods through the entry-point groups `memseg.semantic` and `memseg.segmenters`.

Everything else (reading files, choosing images, writing outputs, scoring against ground truth,
the CLI) is shared and does not know which method runs.

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
│   │                           PairCounts, MethodOutcome, ComparisonItem, MethodSummary, ComparisonSummary,
│   │                           Micrograph, SemanticResult, PixelScores, SemanticSummary, MicrographOutcome
│   ├── ports.py                InstanceSegmenter, ImageSource, MaskReader, GroundTruthReader,
│   │                           ResultSink (+ CompositeSink, NullSink), ReportWriter, ProgressListener,
│   │                           ComparisonSink, ComparisonReportWriter, ComparisonListener,
│   │                           SemanticSegmenter, MicrographReader, GroundTruthMaskReader, SemanticSink,
│   │                           SemanticReportWriter, MicrographListener
│   └── metrics.py              match_instances (Hungarian on IoU), summarize,
│                               sample_pixel_pairs + pair_counts (pair AUROC),
│                               pixel_scores + summarize_semantic (pixel IoU / Dice)
├── application/                CORE — use cases
│   ├── use_cases.py            SegmentImages, EvaluateSegmenter
│   ├── compare.py              CompareSegmenters, PairSampling
│   └── semantic.py             SegmentMicrographs (semantic [-> instance], optional pixel scoring)
├── io/                         adapters for the file system
│   ├── images.py               ImageFileMaskReader, ImageFileGroundTruthReader, write_labels
│   ├── micrographs.py          MicrographFileReader (.mrc + header pixel size, .tif, ...),
│   │                           ImageFileGroundTruthMaskReader, SemanticMaskSink, SemanticPreviewSink,
│   │                           ConsoleMicrographProgress
│   ├── sources.py              FolderImageSource, LabelledDatasetSource (masks/ + labels/)
│   ├── sinks.py                LabelTiffSink, PreviewSink, ArtifactSink
│   ├── previews.py             colour rendering (+ optional skeleton/junction overlay, mono masks)
│   ├── side_by_side.py         SideBySideSink: captioned panels mask | GT | method A | method B ...
│   └── reports.py              CsvReportWriter (summary.csv/.json), ConsoleProgress,
│                               ComparisonCsvReport (comparison.csv/.json), ConsoleComparison (results table)
├── registry.py                 PluginRegistry: built-ins + entry points, lazy import (both families)
├── segmenters/                 adapters: instance methods
│   ├── __init__.py             registry: REGISTRY (built-ins, run-time registration) +
│   │                           entry points in group "memseg.segmenters"
│   ├── base.py                 SegmenterPlugin (CLI options + factory, summary), --config/--set parsing
│   └── components.py           reference method: one instance per connected piece
└── semantic/                   adapters: semantic methods
    ├── __init__.py             registry: REGISTRY + entry points in group "memseg.semantic"
    ├── base.py                 SemanticSegmenterPlugin
    └── threshold.py            reference method: background-corrected dark-line threshold
tests/test_core.py              use cases against fakes (no files, no network)
tests/test_plugins.py           registry, entry-point discovery, reference method
tests/test_semantic.py          pixel metrics, SegmentMicrographs with fakes, threshold, MRC reader, CLI
```

## Core models (`domain/models.py`)

| model | meaning |
|---|---|
| `ImageRef(path, key, gt_path)` | one input; `key` is the relative name outputs mirror (`sub/mask_01.tif` → `OUT/sub/mask_01.tif`) |
| `MembraneMask(data, ref)` | 2-D bool mask, validated on construction |
| `InstanceLabels(data)` | 2-D int32 map, 0 = background; `consecutive()` relabels to 1..N |
| `SegmentationResult(labels, method, info, artifacts, overlay)` | method output. `info`: small scalars for logs/reports (e.g. `scale`, `junctions`); `artifacts`: large optional arrays (e.g. network outputs) for `--save_raw`; `overlay`: optional preview hints (skeleton, junctions), drawn only with `--preview_overlay` |
| `MatchResult`, `EvaluationSummary` | per-image and overall precision / recall / F1 / matched IoU |
| `PairCounts(tp, fn, fp, tn)` | same-instance decisions on sampled pixel pairs; additive over images; `.auroc` |
| `MethodOutcome`, `ComparisonItem` | one method on one image (time, result, match, pairs) / all methods on one image |
| `MethodSummary`, `ComparisonSummary` | per-method totals: setup and run time, instances, F1 etc., pooled pair AUROC |
| `ItemOutcome(ref, status, …)` | what happened to one image (ok / skipped / failed), turns itself into a report row |
| `Micrograph(data, ref, pixel_size)` | 2-D float32 image to segment; `pixel_size` in Å/px or `None` |
| `SemanticResult(mask, method, probability, info, artifacts)` | semantic output: the `MembraneMask` (input of an instance method), optional probability map in [0, 1] |
| `PixelScores(tp, fp, fn, tn)`, `SemanticSummary` | pixel confusion counts (additive; `.iou`, `.dice`, `.precision`, `.recall`) / pooled over images |
| `MicrographOutcome(ref, status, semantic, instances, scores, …)` | one micrograph through `SegmentMicrographs`; report row with `semantic_*` / `instance_*` columns |

## Ports (`domain/ports.py`)

| port | implemented by | purpose |
|---|---|---|
| `InstanceSegmenter.segment(mask) -> SegmentationResult` | `ConnectedComponentsSegmenter`, plugin packages | the method; must do no file I/O |
| `StagedSegmenter` (an `InstanceSegmenter`): `stages()`, `infer(batch)`, `batch_size` | plugin packages (e.g. a network + post-processing) | splits a method into picklable per-image CPU stages and one batchable stage that stays in the owning process; see *Parallel execution* |
| `SegmenterStages.prepare(mask)` / `.finish(mask, prepared, inferred)` | returned by `StagedSegmenter.stages()` | the per-image CPU stages; picklable, no model, no file I/O |
| `ImageSource` (iterable of `ImageRef`) | `FolderImageSource`, `LabelledDatasetSource` | which images; also handles `--sample/--seed/--limit` |
| `MaskReader`, `GroundTruthReader` | `ImageFile*Reader` | load masks / GT labels |
| `ResultSink.exists/save` | `LabelTiffSink`, `PreviewSink`, `ArtifactSink`, `CompositeSink`, `NullSink` | persist results; the first sink of a `CompositeSink` decides "already done" |
| `ReportWriter` | `CsvReportWriter` | `summary.csv` (+ `summary.json` when evaluating) |
| `ProgressListener` | `ConsoleProgress` | log lines; a GUI or plugin host can supply its own |
| `ComparisonSink` | `SideBySideSink` | one side-by-side preview per compared image |
| `ComparisonReportWriter` | `ComparisonCsvReport` | `comparison.csv` (per image) + `comparison.json` (per method) |
| `ComparisonListener` | `ConsoleComparison` | per-image lines and the final results table |
| `SemanticSegmenter.segment(micrograph) -> SemanticResult` | `ThresholdSegmenter`, plugin packages | the semantic method; must do no file I/O |
| `StagedSemanticSegmenter` / `SemanticStages` | plugin packages | the semantic counterpart of `StagedSegmenter` / `SegmenterStages` |
| `MicrographReader`, `GroundTruthMaskReader` | `MicrographFileReader`, `ImageFileGroundTruthMaskReader` | load micrographs (+ pixel size) / GT membrane masks |
| `SemanticSink.exists/save` | `SemanticMaskSink`, `SemanticPreviewSink` (in a `CompositeSink`) | persist masks / probability maps / previews |
| `SemanticReportWriter`, `MicrographListener` | `CsvReportWriter`, `ConsoleMicrographProgress` | `summary.csv` (+ `summary.json` with GT) / log lines |

## Use cases (`application/use_cases.py`)

* **`SegmentImages`**(segmenter, source, reader, sink, report, progress, skip_existing):
  for each image: skip if the sink already has it → read → segment → save. A failure on one image
  is recorded as `FAILED` and the run continues.
* **`EvaluateSegmenter`**(segmenter, source, reader, gt_reader, sink, report, progress, iou_threshold):
  read mask + GT → segment → optionally save → `match_instances` → `summarize`.

  Both take optional `executor`, `batch_size` and `max_in_flight`; see *Parallel execution* below.

* **`CompareSegmenters`**(segmenters: name → method, source, reader, gt_reader, method_sinks,
  comparison_sink, report, progress, setup_seconds, iou_threshold, pairs, warmup):
  per image: read mask and GT (if `ref.gt_path`), sample pixel pairs once, run every method
  (timed with `perf_counter`, segmentation only), score F1 and pair counts when GT exists, save each
  method's labels, then hand all outcomes to the comparison sink. `warmup` runs every method once on
  the first image, untimed, so GPU/cuDNN start-up does not count. Setup time (e.g. loading the network)
  is measured by the composition root and reported separately.

* **`SegmentMicrographs`** (`application/semantic.py`) takes semantic, source, reader, instance,
  semantic_sink, instance_sink, gt_reader, report, progress and skip_existing. For each micrograph it:
  1. reads it and runs the semantic method; a mask whose shape differs from the image's counts as a failure;
  2. saves the mask;
  3. if `instance` is set, runs it on the mask and saves the instances;
  4. if a GT reader is set and the image has `ref.gt_path`, computes `pixel_scores`.

  Pooled scores are returned as a `SemanticSummary`.

None of them imports files, a network library or a concrete method. `tests/test_core.py` and
`tests/test_semantic.py` run them with in-memory fakes.

## Parallel execution (`application/parallel.py`)

`SegmentImages`, `EvaluateSegmenter` and `SegmentMicrographs` share one per-image runner, `run_images`. It takes an
optional `concurrent.futures.Executor`, which the composition root creates (`cli.py`: a
`ProcessPoolExecutor` with the `spawn` start method for `--workers N`). The application layer never
starts processes itself.

| executor | segmenter | what runs where |
|---|---|---|
| none (default) | any | every image in this process, in order: read → `segment` → save (→ match); the behaviour of earlier versions |
| given | plain `InstanceSegmenter` | one job per image in a worker: read → `segment` → save (→ match) |
| given | `StagedSegmenter` | worker: read → `prepare`; **this process**: `infer` on batches of up to `batch_size` prepared images; worker: `finish` → save (→ match) |

Why the split: a network is loaded once, in the process that owns it (the GPU), and is fed in
batches. The per-image CPU work around it (rescaling, post-processing), which is often most of the
time, runs for several images at once. `StagedSegmenter.segment` runs the same three stages in order,
so sequential and parallel runs execute the same code and give identical results.

Rules:
* **Picklable:** everything sent to a worker must be picklable. That covers the reader, sink and
  ground-truth reader, `stages()` (configuration only, never the model) and a plain segmenter.
  Results returned by `prepare` / `infer` must be picklable too.
* **No GPU libraries in workers:** the stages' modules must not import torch or similar at module
  level. Workers are spawned, so they import only what they unpickle.
* **Bounded memory:** at most `max_in_flight` images are read but not yet finished (CLI: 2 × workers).
* **Skips and failures:**
  - `skip_existing` is decided in this process before any work is submitted;
  - a failure in any stage marks only that image `FAILED`. A failed `infer` call fails the images in
    its batch;
  - if the pool breaks (e.g. a worker is killed for running out of memory), the remaining images are
    recorded as failed.
* **Order and timing:**
  - progress is reported in completion order;
  - returned outcomes and `summary.csv` are in input order;
  - `seconds` per image is the sum of its own stage times (prepare + its share of the batch's
    `infer` + finish), so it stays comparable with sequential runs. The console also prints the
    total wall time.

The runner is generic over a per-image *job*, which holds the use case's I/O:
* `ImageJob` (masks): read the mask (+ GT); afterwards save the result and match it against the GT.
* `MicrographJob` (micrographs): read the micrograph. Afterwards it checks the mask shape, saves the
  mask, runs the optional instance method on it and saves that, then scores against a GT mask. In a
  parallel run all of this happens in the worker, so the instance method runs there too.

The method is either the job's plain kind (`InstanceSegmenter` / `SemanticSegmenter`) or its staged
kind (`StagedSegmenter` / `StagedSemanticSegmenter`).

`CompareSegmenters` stays sequential because it measures per-method timing.

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

Shared options: `--method`, `--config JSON`, `--set KEY=VALUE` (repeatable), `--preview_dir`, `--preview_overlay`,
`--limit`, `--sample`, `--seed`, `--workers N` (worker processes; 1 = sequential), `--batch_size N` (`infer` batch of a
staged method); `segment` adds `-o`, `--preview`, `--save_raw`, `--overwrite`,
`--ext`, `--no_recursive`; `evaluate` adds `-o`, `--iou`.

Outputs: `<key>.tif` (uint16, uint32 if > 65 535 instances), `<key>_preview.png`, `<key>_raw.npz`,
`summary.csv` (one row per image: status, size, method info, n_instances, seconds, tp/fp/fn),
`summary.json` (evaluation).

### `semantic`
`memseg semantic MICROGRAPHS... -o OUT [--method SEMANTIC] [--instance METHOD] [--gt_dir DIR]` runs
`SegmentMicrographs`.

* The semantic method's options and the instance method's options are both added to the one parser.
  If two of them clash, the command stops with an error.
* Parameters:
  * semantic method: `--set` / `--config`;
  * instance method: `--instance_set` / `--instance_config`.
* Pixel size: the MRC header value is used (0 and 1.0 count as unset), else `--pixel_size`, else
  `None`, in which case the method uses its own default.
* Outputs:
  * `<key>_mask.tif`;
  * optionally `<key>_prob.tif` and `<key>_mask.png`;
  * `instances/<key>.tif` (+ `_preview.png`);
  * with `--save_masked`, `<key>_masked.mrc` and `instances/<key>_masked.mrc`
    (`MaskedMicrographSink`: re-reads the micrograph with the run's reader, zero outside the mask);
  * `summary.csv`, plus `summary.json` with pooled pixel IoU / Dice when `--gt_dir` is given.
* An existing `<key>_mask.tif` means "done" unless `--overwrite` is given.
* `--workers N` / `--batch_size N`: as for `segment` (see *Parallel execution*). With a staged semantic
  method, its `infer` stays in the main process. Its CPU stages, the instance method, saving and
  scoring run in the workers.

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

**A new semantic method**: the same steps with `SemanticSegmenter` (micrograph → `SemanticResult`),
`memseg.semantic.base.SemanticSegmenterPlugin` and the entry-point group `memseg.semantic`.
Its CLI options share the `memseg semantic` command line with the instance method's, so prefix
them (e.g. `--mynet_model`). A network-based method should be a `StagedSemanticSegmenter`, so that
`--workers` keeps the network in one process.

**A new input/output** (e.g. a CryoSPARC job, a database): implement the matching port
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
