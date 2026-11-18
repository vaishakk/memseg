# memseg

A framework for **membrane segmentation** in cryo-EM, in two stages that can be used together or apart:

* **semantic**: micrograph → **binary membrane mask** (plus an optional probability map);
* **instance**: binary membrane mask → **instance label map**, with one label per membrane, or one
  per layer of a multilamellar vesicle.

`memseg` provides everything around a segmentation method:

* reading micrographs (`.mrc` with header pixel size, `.tif`, ...), mask folders, labelled datasets and ground truth;
* writing label maps, colour previews and CSV/JSON reports;
* scoring:
  * instances: precision, recall, F1 and IoU with Hungarian matching, plus pair AUROC;
  * masks: pixel IoU, Dice, precision and recall;
* side-by-side comparison of methods, with timing;
* a command line interface.

**Methods are plugins.** Any installed package can add one through a Python entry point.
The CLI then offers it next to the built-in reference methods:

* `threshold` (semantic): a background-corrected threshold for dark lines;
* `components` (instance): every connected piece of membrane becomes one instance.

## Installation

Requires Python ≥ 3.11.

```bash
pip install memseg          # or "memseg[mrc]" to read .mrc micrographs
# or, from a clone, with uv
uv sync --extra mrc
```

This installs a `memseg` command; `python -m memseg` works the same way.

## Usage

```bash
memseg methods                                   # list installed methods

# segment masks (files or folders, searched recursively) and write label maps
memseg segment masks/ -o out --method components --preview
memseg segment masks/ -o out --method components --set min_size=200

# score against ground truth: dataset folders containing masks/ and labels/
memseg evaluate data/val --method components --limit 20 -o eval_out

# run several methods on the same masks; compare speed, F1 and pair AUROC
memseg compare data/val -o cmp --methods components my_method --warmup
memseg compare masks/ --gt_dir labels/ -o cmp --methods components my_method --set components.min_size=200

# a method's own options
memseg segment --method components --help

# semantic: micrographs -> membrane masks, optionally -> instances, scored against GT masks
memseg semantic micrographs/ -o out --method threshold --preview
memseg semantic micrographs/ -o out --method threshold --instance components --gt_dir gt_masks/ --save_prob
memseg semantic --method threshold --instance components --help
```

### Common options

| option | meaning |
|---|---|
| `--method`, `-m` | method name (see `memseg methods`); default `components` |
| `--config FILE` / `--set KEY=VALUE` | method parameters from JSON / single overrides (repeatable) |
| `--preview_dir DIR` | write colour PNG previews |
| `--preview_overlay` | draw the method's overlay (skeleton / junctions) on instance previews; off by default |
| `--limit N`, `--sample N`, `--seed S` | process the first N / N random images |
| `--workers N` | process images in N worker processes (default 1 = sequential); see below |
| `--batch_size N` | images per network call of a staged method when `--workers` > 1 |
| `--overwrite` (segment) | recompute images whose output already exists |
| `--save_raw` (segment) | also write method intermediates (`<name>_raw.npz`) |
| `--iou T` (evaluate, compare) | IoU threshold for a match (default 0.5) |

In `compare`, per-method parameters use `--set METHOD.KEY=VALUE` and `--config METHOD=FILE`.
`--methods` defaults to every installed method.

`semantic` has its own options:

| option | meaning |
|---|---|
| `--method`, `-m` | semantic method; default `threshold` |
| `--config` / `--set` | semantic method parameters |
| `--instance METHOD` | also run an instance method on each mask |
| `--instance_config` / `--instance_set` | instance method parameters |
| `--gt_dir DIR` | ground-truth masks with the same relative names: report pixel IoU / Dice |
| `--pixel_size A` | Å/px when the file has none (an MRC header value is used when set; 0 and 1.0 count as unset) |
| `--save_prob`, `--preview` | also write the probability map / PNG previews |
| `--save_masked` | also write the micrograph with everything outside the mask set to 0, as MRC |
| `--workers N`, `--batch_size N` | process micrographs in N worker processes (instance method included); see below |
| `--preview_overlay` | draw the instance method's overlay on instance previews |

### Inputs and outputs

* Inputs: `.tif .tiff .png .jpg .jpeg .bmp` masks (non-zero = membrane).
* Label maps: `<name>.tif` (uint16, uint32 if > 65 535 instances), mirroring the input folder structure.
* `summary.csv` per run (+ `summary.json` for `evaluate`, and for `semantic` with `--gt_dir`).
* `compare` writes:
  * `side_by_side/<name>.png`: mask | GT | one panel per method;
  * `<method>/<name>.tif`;
  * `comparison.csv` and `comparison.json`.
* `semantic` writes:
  * `<name>_mask.tif` (uint8, 255 = membrane), plus `<name>_prob.tif` and `<name>_mask.png` when asked;
  * `instances/<name>.tif` (+ `_preview.png`) with `--instance`;
  * with `--save_masked`: `<name>_masked.mrc`, the micrograph (float32, header pixel size) with every
    pixel outside the membrane mask set to 0, and with `--instance` also `instances/<name>_masked.mrc`,
    the same for the kept instances.

## Performance: `--workers`

`--workers N` processes N images at once in worker processes. Each image is still processed on one
core, so a single image doesn't get faster; a folder of images does. The output is **byte-identical**
to a sequential run: every file below was compared with the 1-worker run.

| command | images | 1 worker | 4 | 8 | 12 | speed-up (best) |
|---|---|---|---|---|---|---|
| `segment --method components` | 48 masks | 20.9 s | 9.9 s | 9.4 s | 8.1 s | 2.6× |
| `semantic --method threshold --instance components` | 24 micrographs | 62.5 s | 23.4 s | 16.6 s | 17.8 s | 3.8× |

**Reading the numbers**

* **How much a method gains depends on its CPU work per image.**
  * `components` takes about 0.4 s per image, and much of that is reading and writing files. That part
    can't run any faster in parallel.
  * `threshold` + `components` takes about 2.6 s of computation per micrograph, so it gains more.
  * Heavier methods, with several seconds of CPU post-processing per image, reached 4.5–6× with 8–12
    workers on the same machine.
* **Network methods:** for a staged method (a network + post-processing), the network stage runs in one
  process and limits the speed-up. A method whose network takes about a third of the time gained
  about 2×.

**Memory: how many workers fit**

Peak memory per process. `segment` was measured on 16 masks (8 × 1919×1855 + 8 × 4092×5760) and
`semantic` on 12 micrographs (4092×5760).

| command | main process | per worker | total at 1 / 4 / 8 / 12 workers |
|---|---|---|---|
| `segment --method components` | 0.8–0.9 GB (1.2 GB at 1 worker) | 0.6 GB | 1.2 / 2.3 / 4.4 / 4.7 GB |
| `semantic --method threshold --instance components` | 2.5–2.6 GB (3.0 GB at 1 worker) | 0.8 GB | 3.0 / 4.2 / 5.3 / 8.4 GB |

To choose `--workers` for your machine:

```
RAM needed ≈ main process + N × per worker + results kept by the main process
```

* **Treat the per-worker figure as an upper bound.** It is the peak RSS of the largest worker,
  including shared libraries. Measured totals are lower.
* **Leave headroom** for the OS, and don't use more workers than physical cores.
* **Memory per worker scales with the pixel count of the largest images.** The figures are for
  4092×5760 (23.6 Mpx) images.
* **Methods add their own memory.** A method with a network also needs its model in the main
  process, plus GPU memory. GPU memory does not grow with `--workers`, because only the main process
  uses the GPU.
* **Results kept by the main process (current limitation):** the use cases return every image's
  result in memory, label maps included, with or without `--workers`. The main process therefore grows
  per processed image by about:
  * `segment`: 4 bytes/pixel (label map);
  * `semantic`: 9 bytes/pixel (mask + probability map + instance labels).

  For a 4092×5760 image that is about 94 MB or 212 MB. For large folders, process them in parts
  (`--limit`, or one sub-folder at a time; existing outputs are skipped).
* **How memory was measured:** peaks sampled every 0.25 s from `/proc/<pid>/smaps_rollup` for the
  `memseg` process and its children. The main process and per-worker figures are RSS; the total is
  PSS summed over all processes, so shared pages are counted once.

**Conditions**

* **Hardware:** AMD Ryzen 9 3900XT (12 cores / 24 threads), 62 GB RAM.
* **Software:**
  * Ubuntu 24.04, Linux 7.0;
  * Python 3.11.15, numpy 2.4.6, scipy 1.17.1, scikit-image 0.26.0;
  * memseg 0.3.0.dev0.
* **A desktop in normal use, not an isolated machine:** about 22 GB of RAM was in use by other programs.
* **Inputs:**
  * Masks: 24 copies each of a 1919×1855 and a 4092×5760 binary membrane mask (TIFF).
  * Micrographs: 24 copies of one 4092×5760 cryo-EM micrograph (MRC, 1.737 Å/px).
  * Every run used copies of the same files. Inputs and outputs were on a local SSD, and the files were
    in the page cache after the first run.
* **Settings:** default method parameters.
* **Timing:** the wall time of the whole command, including start-up, starting the workers and writing
  all outputs (label maps, masks, `summary.csv`). Each configuration was run **once**, so expect a few
  percent of variation.
* **Memory:** measured per process, as described above, in a separate run on the same machine and
  files.
* **Workers:** `spawn` start method, one OpenMP/OpenBLAS/MKL thread per worker. `memseg` sets these
  thread limits unless you have set them yourself.

## Writing your own method

A segmentation method is any class that implements `InstanceSegmenter.segment(mask) -> SegmentationResult`.
The framework handles everything else. A method must not do file I/O.

### 1. Implement the method

The example below closes small gaps in the membrane before labelling its connected pieces:

```python
# my_method/segmenter.py
import numpy as np
from scipy import ndimage

from memseg.domain import InstanceLabels, InstanceSegmenter, MembraneMask, SegmentationResult
from memseg.segmenters.base import SegmenterPlugin


class GapClosingSegmenter(InstanceSegmenter):
    """Bridge gaps up to `gap` px, then one instance per connected piece."""

    name = "gapclose"

    def __init__(self, gap: int = 3):
        self.gap = gap

    def describe(self) -> str:  # shown in logs
        return f"gapclose (gap={self.gap})"

    def segment(self, mask: MembraneMask) -> SegmentationResult:  # mask.data: 2-D bool
        closed = ndimage.binary_closing(mask.data, np.ones((2 * self.gap + 1,) * 2))
        labels, n = ndimage.label(closed, structure=np.ones((3, 3)))
        labels[~mask.data] = 0  # keep only original membrane pixels
        return SegmentationResult(
            labels=InstanceLabels.consecutive(labels),  # relabel 1..N
            method=self.name,
            info=dict(pieces=n),  # small scalars -> summary.csv column
        )


class GapClosingPlugin(SegmenterPlugin):
    """Connects the method to the CLI: its options and how to build it."""

    name = "gapclose"
    summary = "close small gaps, then connected components"

    def add_arguments(self, group):
        group.add_argument("--gap", type=int, default=3, help="largest gap to bridge (px)")

    def build(self, args, overrides):  # overrides = --config JSON + --set key=value
        return GapClosingSegmenter(gap=overrides.get("gap", args.gap))
```

`SegmentationResult` can also carry `artifacts` (large arrays, written by `--save_raw`) and an
`overlay` (skeleton / junctions drawn on previews with `--preview_overlay`).

### 2. Register it with the CLI

**As an installable package (recommended).** Declare an entry point in your package's `pyproject.toml`:

```toml
[project]
name = "my-method"
dependencies = ["memseg>=0.2"]

[project.entry-points."memseg.segmenters"]
gapclose = "my_method.segmenter:GapClosingPlugin"
```

After `pip install -e .`, the method is available everywhere:

```bash
memseg methods
memseg segment masks/ -o out --method gapclose --gap 5 --preview
memseg compare data/val -o cmp --methods components gapclose
```

The plugin module is imported only when the method is used. A method that needs PyTorch therefore
does not slow down or break the other methods. Your package can be private: the framework only needs it installed.

**At run time, without packaging.**

```python
import sys
from memseg import segmenters
from memseg.cli import main

segmenters.REGISTRY["gapclose"] = "my_method.segmenter:GapClosingPlugin"
sys.exit(main())
```

### 3. Or use it from Python with the built-in use cases and adapters

```python
from memseg.application import EvaluateSegmenter, SegmentImages
from memseg.domain import CompositeSink
from memseg.io import (
    ConsoleProgress,
    CsvReportWriter,
    FolderImageSource,
    ImageFileGroundTruthReader,
    ImageFileMaskReader,
    LabelledDatasetSource,
    LabelTiffSink,
    PreviewSink,
)

from my_method.segmenter import GapClosingSegmenter

seg = GapClosingSegmenter(gap=5)

# segment: label maps + previews + summary.csv in out/
SegmentImages(
    segmenter=seg,
    source=FolderImageSource(["masks/"]),  # files or folders
    reader=ImageFileMaskReader(),
    sink=CompositeSink([LabelTiffSink("out"), PreviewSink("out")]),
    report=CsvReportWriter("out"),
    progress=ConsoleProgress(),
).run()

# evaluate against ground truth (dataset folders with masks/ and labels/)
outcomes, summary = EvaluateSegmenter(
    segmenter=seg,
    source=LabelledDatasetSource(["data/val"], limit=20),
    reader=ImageFileMaskReader(),
    gt_reader=ImageFileGroundTruthReader(),
    progress=ConsoleProgress(),
).run()
print(summary.f1, summary.mean_matched_iou)
```

You can swap any adapter for your own implementation of the matching port in `memseg.domain`
(`ImageSource`, `MaskReader`, `ResultSink`, `ReportWriter`, `ProgressListener`). For example, a port
could read MRC files or write results into a CryoSPARC job.

### Network + post-processing: `StagedSegmenter`

`--workers N` runs images in N worker processes. A plain method runs whole in a worker. A method
built around a network should not load the model in every worker. Split it with `StagedSegmenter`
instead:

* `infer` stays in the main process with the model and receives batches;
* `prepare` / `finish`, the per-image CPU work, run in the workers.

```python
from dataclasses import dataclass

import numpy as np
from memseg.domain import InstanceLabels, SegmentationResult, SegmenterStages, StagedSegmenter


@dataclass
class MyStages(SegmenterStages):  # picklable: configuration only, no model
    threshold: float = 0.5

    def prepare(self, mask):  # worker: before the network
        return mask.data.astype(np.float32)

    def finish(self, mask, prepared, prob):  # worker: post-processing
        labels = my_postprocess(prob > self.threshold)
        return SegmentationResult(labels=InstanceLabels(labels), method="mynet")


class MyNetSegmenter(StagedSegmenter):
    name = "mynet"
    batch_size = 4  # preferred images per infer() call

    def __init__(self, model):
        self.model = model

    def stages(self):
        return MyStages()

    def infer(self, prepared):  # main process: one result per input, in order
        return [run_network(self.model, x) for x in prepared]
```

`segment(mask)` runs `prepare → infer → finish`, so the method still works sequentially and gives
the same results either way. Semantic methods use the same pattern: `StagedSemanticSegmenter` with
`SemanticStages`, and `memseg semantic --workers N`.
Keep GPU imports out of the module that defines the stages: workers
are spawned and import only what they unpickle. Details are in
[architecture.md](https://github.com/vaishakk/memseg/blob/main/architecture.md).

### Semantic methods

A semantic method implements `SemanticSegmenter.segment(image: Micrograph) -> SemanticResult`.

* `image.data` is a 2-D float32 array, and `image.pixel_size` is in Å/px, or `None` if unknown.
* The result holds:
  * the `MembraneMask`;
  * an optional `probability` map in [0, 1];
  * `info` scalars, which go into `summary.csv`.
* Its plugin subclasses `memseg.semantic.base.SemanticSegmenterPlugin` and registers in its own
  entry-point group.

Its CLI options share the command line with the instance method's, so prefix them, e.g. `--mynet_model`.

```python
from memseg.domain import MembraneMask, SemanticResult, SemanticSegmenter
from memseg.semantic.base import SemanticSegmenterPlugin


class MyNet(SemanticSegmenter):
    name = "mynet"

    def segment(self, image):
        prob = run_my_network(image.data, image.pixel_size)  # your model
        return SemanticResult(MembraneMask(prob > 0.5), self.name, probability=prob)


class MyNetPlugin(SemanticSegmenterPlugin):
    name = "mynet"
    summary = "my membrane network"

    def build(self, args, overrides):
        return MyNet()
```

```toml
[project.entry-points."memseg.semantic"]
mynet = "my_method.semantic:MyNetPlugin"
```

From Python, `SegmentMicrographs` chains any semantic method with any instance method:

```python
from memseg.application import SegmentMicrographs
from memseg.io import ConsoleMicrographProgress, FolderImageSource, MicrographFileReader, SemanticMaskSink
from memseg.segmenters.components import ConnectedComponentsSegmenter
from memseg.semantic.threshold import ThresholdSegmenter

outcomes, summary = SegmentMicrographs(
    semantic=ThresholdSegmenter(),
    instance=ConnectedComponentsSegmenter(min_size=2000),  # optional
    source=FolderImageSource(["micrographs/"], exts=[".mrc"]),
    reader=MicrographFileReader(default_pixel_size=2.7),
    semantic_sink=SemanticMaskSink("out", probability=True),
    progress=ConsoleMicrographProgress(),
).run()
```

## Project layout

```
memseg/
├── cli.py            command line / composition root
├── registry.py       plugin discovery (built-ins + entry points), shared by both method families
├── domain/           models, ports (interfaces), metrics — no I/O
├── application/      use cases: SegmentImages, EvaluateSegmenter, CompareSegmenters, SegmentMicrographs
├── io/               file readers/writers (incl. micrographs), previews, reports
├── segmenters/       instance methods: registry, SegmenterPlugin, `components`
└── semantic/         semantic methods: registry, SemanticSegmenterPlugin, `threshold`
tests/                use-case tests with in-memory fakes, plugins, semantic pipeline, CLI
```

See [architecture.md](https://github.com/vaishakk/memseg/blob/main/architecture.md) for the design, the pair-AUROC metric and how to add
methods or I/O adapters.

## Development

```bash
uv sync --extra mrc
uv run pytest              # tests
uv run ruff check .        # lint
uv run ruff format .       # format
uv run mypy                # type-check
```

CI runs all four on every push and pull request.

## License

MIT, see [LICENSE](https://github.com/vaishakk/memseg/blob/main/LICENSE).
