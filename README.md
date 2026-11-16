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
  * `instances/<name>.tif` (+ `_preview.png`) with `--instance`.

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
the same results either way. Keep GPU imports out of the module that defines the stages: workers
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
