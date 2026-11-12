"""Plugin contract for semantic segmentation methods.

Same shape as memseg.segmenters.base.SegmenterPlugin: declare CLI options, build a configured
`SemanticSegmenter`. `overrides` merges --semantic_config JSON and --semantic_set key=value pairs.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Any

from ..domain.ports import SemanticSegmenter


class SemanticSegmenterPlugin(ABC):
    name: str = ""
    #: one line shown by `memseg methods`
    summary: str = ""

    def add_arguments(self, group: argparse._ArgumentGroup) -> None:
        """Declare method-specific CLI options (optional). Prefix them to avoid clashes with
        the instance method's options, which share the command line."""

    @abstractmethod
    def build(self, args: argparse.Namespace, overrides: dict[str, Any]) -> SemanticSegmenter:
        """Create the segmenter."""
