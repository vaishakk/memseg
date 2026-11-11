"""memseg: membrane instance segmentation with interchangeable segmentation methods.

Layers (dependencies point inwards only, see architecture.md):
    memseg.domain       models, ports (interfaces) and metrics; numpy only
    memseg.application  use cases that orchestrate ports
    memseg.segmenters   segmentation methods, each implementing domain.ports.InstanceSegmenter
    memseg.io           file-system adapters (readers, writers, sources, reports)
    memseg.cli          composition root: parses arguments and wires everything together
"""

__version__ = "0.2.0"
