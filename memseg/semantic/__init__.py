"""Registry of semantic segmentation methods (micrograph -> membrane mask).

Methods come from ``REGISTRY`` (built-ins and run-time registration) and from installed
packages that declare an entry point in the ``memseg.semantic`` group::

    [project.entry-points."memseg.semantic"]
    mynet = "mypackage.semantic:MyNetPlugin"

Plugins are imported lazily, so a method's heavy dependencies load only when it is used.
"""

from __future__ import annotations

from ..registry import PluginRegistry
from .base import SemanticSegmenterPlugin

ENTRY_POINT_GROUP = "memseg.semantic"

# name -> "module:PluginClass"
REGISTRY: dict[str, str] = {
    "threshold": "memseg.semantic.threshold:ThresholdPlugin",
}

SUMMARIES: dict[str, str] = {}

_registry = PluginRegistry(ENTRY_POINT_GROUP, REGISTRY, SUMMARIES)


def available() -> list[str]:
    return _registry.available()


def get_plugin(name: str) -> SemanticSegmenterPlugin:
    return _registry.get_plugin(name)


def summary(name: str) -> str:
    return _registry.summary(name)
