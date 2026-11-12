"""Plugin registry shared by the instance and semantic method families.

A registry merges two sources of `name -> "module:PluginClass"`:

* `builtins`: methods shipped with memseg, plus any registered at run time;
* installed packages declaring an entry point in the registry's group.

Plugin modules are imported lazily (only in `get_plugin`), so a method's heavy dependencies
load only when that method is used. A built-in name wins over an entry point with the same name.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Any


class PluginRegistry:
    def __init__(self, group: str, builtins: dict[str, str], summaries: dict[str, str] | None = None):
        self.group = group
        self.builtins = builtins          # mutable: REGISTRY[name] = "module:Class" registers at run time
        self.summaries = summaries if summaries is not None else {}

    def _entry_points(self) -> dict[str, str]:
        # looked up through the module so tests can monkeypatch memseg.registry.entry_points
        from . import registry as _self
        return {ep.name: ep.value for ep in _self.entry_points(group=self.group)}

    def targets(self) -> dict[str, str]:
        return {**self.builtins, **{k: v for k, v in self._entry_points().items() if k not in self.builtins}}

    def available(self) -> list[str]:
        return list(self.targets())

    def get_plugin(self, name: str) -> Any:
        targets = self.targets()
        if name not in targets:
            raise KeyError(f"unknown method {name!r}; available: {', '.join(targets)}")
        module, cls = targets[name].split(":")
        return getattr(importlib.import_module(module), cls)()

    def summary(self, name: str) -> str:
        if name in self.summaries:
            return self.summaries[name]
        try:
            return self.get_plugin(name).summary
        except Exception as exc:  # a broken plugin must not break `memseg methods`
            return f"(failed to load: {exc})"


__all__ = ["PluginRegistry", "entry_points"]
