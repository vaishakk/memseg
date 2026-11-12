"""Plugin contract for segmentation methods.

A method is made available to the CLI by a `SegmenterPlugin`: it declares its command-line
options and builds a configured `InstanceSegmenter`. Plugins are registered by name in
memseg.segmenters (lazy import, so heavy dependencies load only when the method is used).
"""

from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from typing import Any

from ..domain.ports import InstanceSegmenter


class SegmenterPlugin(ABC):
    name: str = ""
    #: one line shown by `memseg methods`
    summary: str = ""

    def add_arguments(self, group: argparse._ArgumentGroup) -> None:
        """Declare method-specific CLI options (optional)."""

    @abstractmethod
    def build(self, args: argparse.Namespace, overrides: dict[str, Any]) -> InstanceSegmenter:
        """Create the segmenter. `overrides` merges --config JSON and --set key=value pairs."""


def parse_value(text: str) -> Any:
    """'3' -> 3, '0.5' -> 0.5, 'true' -> True, 'abc' -> 'abc' (JSON where possible)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_overrides(config_path: str | None, pairs: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if config_path:
        with open(config_path) as f:
            out.update(json.load(f))
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--set expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = parse_value(v.strip())
    return out
