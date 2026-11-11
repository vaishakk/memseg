"""File-system adapters implementing the domain ports."""

from .images import ImageFileGroundTruthReader, ImageFileMaskReader, read_image, to_binary, write_labels
from .reports import ComparisonCsvReport, ConsoleComparison, ConsoleProgress, CsvReportWriter
from .side_by_side import SideBySideSink
from .sinks import ArtifactSink, LabelTiffSink, PreviewSink
from .sources import IMAGE_EXTS, FolderImageSource, LabelledDatasetSource

__all__ = [
    "ImageFileGroundTruthReader", "ImageFileMaskReader", "read_image", "to_binary", "write_labels",
    "ComparisonCsvReport", "ConsoleComparison", "SideBySideSink", "ConsoleProgress", "CsvReportWriter",
    "ArtifactSink", "LabelTiffSink", "PreviewSink", "IMAGE_EXTS",
    "FolderImageSource", "LabelledDatasetSource",
]
