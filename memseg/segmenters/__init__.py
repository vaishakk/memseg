"""Registry of segmentation methods.

Methods come from two places:

* ``REGISTRY``: built-in methods, plus any registered at run time
  (``REGISTRY["name"] = "module:PluginClass"``).
* Installed packages that declare an entry point in the ``memseg.segmenters`` group::

      [project.entry-points."memseg.segmenters"]
      mymethod = "mypackage.segmenter:MyPlugin"

Plugins are imported lazily, so a method's heavy dependencies (e.g. torch) load only
when that method is used.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from .base import SegmenterPlugin

ENTRY_POINT_GROUP = "memseg.segmenters"

# name -> "module:PluginClass"
REGISTRY: dict[str, str] = {
    "components": "memseg.segmenters.components:ConnectedComponentsPlugin",
}

# optional one-line descriptions for `memseg methods` (else the plugin's `summary`)
SUMMARIES: dict[str, str] = {}


def _entry_points() -> dict[str, str]:
    return {ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)}


def _all() -> dict[str, str]:
    return {**REGISTRY, **{k: v for k, v in _entry_points().items() if k not in REGISTRY}}


def available() -> list[str]:
    return list(_all())


def get_plugin(name: str) -> SegmenterPlugin:
    targets = _all()
    if name not in targets:
        raise KeyError(f"unknown method {name!r}; available: {', '.join(targets)}")
    module, cls = targets[name].split(":")
    return getattr(importlib.import_module(module), cls)()


def summary(name: str) -> str:
    if name in SUMMARIES:
        return SUMMARIES[name]
    try:
        return get_plugin(name).summary
    except Exception as exc:  # a broken plugin must not break `memseg methods`
        return f"(failed to load: {exc})"
