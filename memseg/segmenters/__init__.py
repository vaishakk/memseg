"""Registry of instance segmentation methods (binary mask -> instance labels).

Methods come from two places:

* ``REGISTRY``: built-in methods, plus any registered at run time
  (``REGISTRY["name"] = "module:PluginClass"``).
* Installed packages that declare an entry point in the ``memseg.segmenters`` group::

      [project.entry-points."memseg.segmenters"]
      mymethod = "mypackage.segmenter:MyPlugin"

Plugins are imported lazily, so a method's heavy dependencies (e.g. torch) load only
when that method is used. Semantic methods (micrograph -> mask) live in memseg.semantic.
"""

from __future__ import annotations

from ..registry import PluginRegistry
from .base import SegmenterPlugin

ENTRY_POINT_GROUP = "memseg.segmenters"

# name -> "module:PluginClass"
REGISTRY: dict[str, str] = {
    "components": "memseg.segmenters.components:ConnectedComponentsPlugin",
}

# optional one-line descriptions for `memseg methods` (else the plugin's `summary`)
SUMMARIES: dict[str, str] = {}

_registry = PluginRegistry(ENTRY_POINT_GROUP, REGISTRY, SUMMARIES)


def available() -> list[str]:
    return _registry.available()


def get_plugin(name: str) -> SegmenterPlugin:
    return _registry.get_plugin(name)


def summary(name: str) -> str:
    return _registry.summary(name)
