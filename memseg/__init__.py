"""memseg: membrane segmentation (semantic and instance) with interchangeable methods.

Layers (dependencies point inwards only, see architecture.md):
    memseg.domain       models, ports (interfaces) and metrics; numpy only
    memseg.application  use cases that orchestrate ports
    memseg.segmenters   instance methods (registry + plugins), implementing domain.ports.InstanceSegmenter
    memseg.semantic     semantic methods (registry + plugins), implementing domain.ports.SemanticSegmenter
    memseg.io           file-system adapters (readers, writers, sources, reports)
    memseg.cli          composition root: parses arguments and wires everything together
"""

__version__ = "0.3.0.dev0"
