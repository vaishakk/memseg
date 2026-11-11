# memseg

A framework for **membrane instance segmentation** in cryo-EM. It turns **binary membrane masks** into
**instance label maps**: one label per membrane, or one per layer of a multilamellar vesicle.

`memseg` provides everything around a segmentation method:

* reading mask folders, labelled datasets and ground truth;
* writing label maps, colour previews and CSV/JSON reports;
* scoring: precision, recall, F1 and IoU with Hungarian matching, plus pair AUROC;
* side-by-side comparison of methods, with timing;
* a command line interface.

**Methods are plugins.** Any installed package can add one through a Python entry point.
The CLI then offers it next to the built-in reference method `components`, which makes every
connected piece of membrane one instance.

## Installation

Requires Python ≥ 3.11.

```bash
pip install git+https://github.com/vaishakk/memseg.git
# or, from a clone, with uv
uv sync
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
```

### Common options

| option | meaning |
|---|---|
| `--method`, `-m` | method name (see `memseg methods`); default `components` |
| `--config FILE` / `--set KEY=VALUE` | method parameters from JSON / single overrides (repeatable) |
| `--preview_dir DIR` | write colour PNG previews |
| `--limit N`, `--sample N`, `--seed S` | process the first N / N random images |
| `--overwrite` (segment) | recompute images whose output already exists |
| `--save_raw` (segment) | also write method intermediates (`<name>_raw.npz`) |
| `--iou T` (evaluate, compare) | IoU threshold for a match (default 0.5) |

In `compare`, per-method parameters use `--set METHOD.KEY=VALUE` and `--config METHOD=FILE`.
`--methods` defaults to every installed method.

### Inputs and outputs

* Inputs: `.tif .tiff .png .jpg .jpeg .bmp` masks (non-zero = membrane).
* Label maps: `<name>.tif` (uint16, uint32 if > 65 535 instances), mirroring the input folder structure.
* `summary.csv` per run (+ `summary.json` for `evaluate`).
* `compare` writes:
  * `side_by_side/<name>.png`: mask | GT | one panel per method;
  * `<method>/<name>.tif`;
  * `comparison.csv` and `comparison.json`.

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

    def describe(self) -> str:                                   # shown in logs
        return f"gapclose (gap={self.gap})"

    def segment(self, mask: MembraneMask) -> SegmentationResult:  # mask.data: 2-D bool
        closed = ndimage.binary_closing(mask.data, np.ones((2 * self.gap + 1,) * 2))
        labels, n = ndimage.label(closed, structure=np.ones((3, 3)))
        labels[~mask.data] = 0                                   # keep only original membrane pixels
        return SegmentationResult(
            labels=InstanceLabels.consecutive(labels),           # relabel 1..N
            method=self.name,
            info=dict(pieces=n),                                 # small scalars -> summary.csv column
        )


class GapClosingPlugin(SegmenterPlugin):
    """Connects the method to the CLI: its options and how to build it."""

    name = "gapclose"
    summary = "close small gaps, then connected components"

    def add_arguments(self, group):
        group.add_argument("--gap", type=int, default=3, help="largest gap to bridge (px)")

    def build(self, args, overrides):          # overrides = --config JSON + --set key=value
        return GapClosingSegmenter(gap=overrides.get("gap", args.gap))
```

`SegmentationResult` can also carry `artifacts` (large arrays, written by `--save_raw`) and an
`overlay` (skeleton / junctions drawn on previews).

### 2. Register it with the CLI

**As an installable package (recommended).** Declare an entry point in your package's `pyproject.toml`:

```toml
[project]
name = "my-method"
dependencies = ["memseg @ git+https://github.com/vaishakk/memseg.git"]

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
from memseg.io import (ConsoleProgress, CsvReportWriter, FolderImageSource, ImageFileGroundTruthReader,
                       ImageFileMaskReader, LabelledDatasetSource, LabelTiffSink, PreviewSink)

from my_method.segmenter import GapClosingSegmenter

seg = GapClosingSegmenter(gap=5)

# segment: label maps + previews + summary.csv in out/
SegmentImages(
    segmenter=seg,
    source=FolderImageSource(["masks/"]),                    # files or folders
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

## Project layout

```
memseg/
├── cli.py            command line / composition root
├── domain/           models, ports (interfaces), metrics — no I/O
├── application/      use cases: SegmentImages, EvaluateSegmenter, CompareSegmenters
├── io/               file readers/writers, previews, reports
└── segmenters/       plugin registry (built-ins + entry points), SegmenterPlugin, `components`
tests/                use-case tests with in-memory fakes, plugin registry tests
```

See [architecture.md](architecture.md) for the design, the pair-AUROC metric and how to add
methods or I/O adapters.

## Tests

```bash
uv run pytest
```

## License

MIT, see [LICENSE](LICENSE).
