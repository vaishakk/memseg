# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

`memseg` is a framework for membrane segmentation in cryo-EM:
* **semantic** segmentation: micrograph → membrane mask;
* **instance** segmentation: mask → labelled membranes.

Methods are plugins. Running and scoring them is done by the framework.

## Read `architecture.md` first

Read [architecture.md](architecture.md) before you change anything. It defines the layers, models, ports, use cases and extension points. Changes must fit that design and must not work around it.

## Respect the architecture boundaries

The code is split into layers (ports and adapters), and imports only point **inwards**:

| layer | may import | must not import |
|---|---|---|
| `memseg/domain/` (models, ports, metrics) | numpy / scipy only | anything else in `memseg` |
| `memseg/application/` (use cases) | `domain` | `io`, `segmenters`, `semantic`, `cli` |
| `memseg/io/` (file adapters) | `domain` | `application`, `segmenters`, `semantic` |
| `memseg/segmenters/`, `memseg/semantic/` (registries + methods) | `domain`, `registry`, their own `base` | `application`, `io`, each other's methods |
| `memseg/cli.py` (composition root) | everything | — |

Rules that follow from this:
* **Use cases only see ports.** They don't know about files, networks or any particular method. New behaviour that needs I/O goes in a port (`domain/ports.py`) and an adapter (`io/`). The adapter is wired up in `cli.py`.
* **Methods do no file I/O.**
  - A semantic method takes a `Micrograph` and returns a `SemanticResult`.
  - An instance method takes a `MembraneMask` and returns a `SegmentationResult`.
  - Put diagnostics in `info` and large arrays in `artifacts`.
* **The two stages share only a `MembraneMask`.** Don't pass method-specific state between them.
* **Heavy dependencies stay at the edge.**
  - Import torch and similar libraries inside the method that needs them, never in `domain`, `application`, `io` or a registry module.
  - Plugins are imported lazily (`get_plugin`), so listing methods must not import any of them.
* **Built-in methods stay lightweight.**
  - The built-ins are `components` and `threshold`. They are reference methods that need only numpy and scipy.
  - Real methods (networks, private algorithms) live in their own packages. They register through the entry-point groups `memseg.segmenters` and `memseg.semantic`.
* **Parallelism is injected.** Use cases take an optional `concurrent.futures.Executor` created by the composition root. Never start processes or threads inside `domain`, `application` or a method. A `StagedSegmenter`'s `stages()` and everything sent to workers must be picklable, and stage modules must not import torch or other GPU libraries at module level.
* **Scoring lives in `domain/metrics.py`** so that every method is scored the same way. Don't compute metrics inside a method.

If a change seems to need a boundary violation, the abstraction is probably missing something. Discuss it or extend the port. Don't add a cross-layer import.

## Keep the public repo free of private code

This repo is public. Don't add, name or link private methods, their weights or their data:
* no method-specific code, comments or docs;
* no local paths;
* no checkpoints (`*.pt`, `*.sav`) or micrographs.

Examples and tests use synthetic data or the built-in methods.

## Setup and commands

The project uses [uv](https://docs.astral.sh/uv/) and needs Python ≥ 3.11.

```bash
uv sync                      # install (add --extra mrc for .mrc micrographs)
uv run pytest                # run the test suite
uv run ruff check --fix .    # lint (config in pyproject.toml)
uv run ruff format .         # format
uv run mypy                  # type-check memseg/
uv run memseg methods        # list installed instance + semantic methods
uv run memseg segment masks/ -o out --preview
uv run memseg semantic micrographs/ -o out --instance components --preview
```

## Making changes

* **Checks:** `ruff check`, `ruff format --check`, `mypy` and `pytest` must all pass before you finish (CI runs the same). Don't silence a finding with `# noqa` / `# type: ignore` unless the tool is wrong; then add the reason on the same line.
* **Tests:**
  - New behaviour needs a test. Use cases are tested with in-memory fakes of the ports (see `tests/test_core.py` and `tests/test_semantic.py`), not with real files or models.
  - Plugin discovery is tested by monkeypatching `memseg.registry.entry_points`.
* **Tests stay fast and self-contained:**
  - They use synthetic images and `tmp_path`.
  - They don't download anything, need a GPU or read data outside the repo.
  - Optional dependencies are guarded with `pytest.importorskip`.
* **Keep the docs in sync.** Update `README.md` (user-facing usage, CLI options) and `architecture.md` (models, ports, use cases, extension points) in the same change as the code they describe.
* **Public API:**
  - Names exported from `memseg.domain`, `memseg.application` and `memseg.io`, the plugin base classes, the entry-point group names and the CLI flags are the public API. Changing them breaks external method packages.
  - Avoid breaking changes. If one is unavoidable, call it out.
* **New models:** domain models are dataclasses that validate their inputs in `__post_init__`, e.g. mask/image shape and dtype. Follow that pattern.
* **CLI options:**
  - Each method's options are added by its plugin (`add_arguments`).
  - Semantic and instance options share one command line, so plugin options need a method prefix (e.g. `--mynet_model`).
  - Generic parameters go through `--set key=value` / `--config file.json`, and unknown keys are rejected.
* **Style:**
  - Match the existing code: type hints, `from __future__ import annotations`, short docstrings that explain *why*, and no new dependencies without a good reason.
  - Keep functions small and side-effect free in `domain`.
* **Generated files:** don't commit weights, image data, run outputs (`out/`, `*.tif`) or `.venv`. `.gitignore` already covers them.
* **Git:**
  - Don't commit, tag or push unless asked.
  - Never rewrite published history.
