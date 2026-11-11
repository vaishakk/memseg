"""Core tests: use cases run against fakes, without files, torch or a real method.

Run:  python -m pytest tests   (or: python tests/test_core.py)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memseg.application import CompareSegmenters, EvaluateSegmenter, SegmentImages  # noqa: E402
from memseg.domain import (GroundTruthReader, ImageRef, ImageSource, InstanceLabels, InstanceSegmenter,  # noqa: E402
                           MaskReader, MembraneMask, PairCounts, ResultSink, SegmentationResult, Status,
                           match_instances, pair_counts, sample_pixel_pairs)


def two_rings():
    yy, xx = np.mgrid[:100, :200]
    r1 = np.abs(np.hypot(yy - 50, xx - 50) - 30) < 2
    r2 = np.abs(np.hypot(yy - 50, xx - 150) - 30) < 2
    gt = np.zeros((100, 200), np.int32)
    gt[r1], gt[r2] = 1, 2
    return r1 | r2, gt


class ConnectedComponents(InstanceSegmenter):
    """Trivial method: each connected piece of membrane is one instance."""
    name = "cc"

    def segment(self, mask):
        from scipy import ndimage as ndi
        lab, _ = ndi.label(mask.data, structure=np.ones((3, 3)))
        return SegmentationResult(InstanceLabels(lab), self.name, info=dict(pieces=int(lab.max())))


class Broken(InstanceSegmenter):
    name = "broken"

    def segment(self, mask):
        raise RuntimeError("boom")


class Memory(ImageSource, MaskReader, GroundTruthReader):
    def __init__(self, items):  # key -> (mask, gt)
        self.items = items

    def __iter__(self):
        return iter(ImageRef(k, k, gt_path=k) for k in self.items)

    def read(self, ref):
        raise NotImplementedError


class MaskMem(MaskReader):
    def __init__(self, mem):
        self.mem = mem

    def read(self, ref):
        return MembraneMask(self.mem.items[ref.key][0], ref)


class GtMem(GroundTruthReader):
    def __init__(self, mem):
        self.mem = mem

    def read(self, ref):
        return InstanceLabels(self.mem.items[ref.key][1])


class ListSink(ResultSink):
    def __init__(self):
        self.saved = {}

    def exists(self, ref):
        return ref.key in self.saved

    def save(self, ref, result):
        self.saved[ref.key] = result


def test_segment_images_uses_injected_method_and_sink():
    mask, _ = two_rings()
    mem = Memory({"a": (mask, None), "b": (mask, None)})
    sink = ListSink()
    out = SegmentImages(ConnectedComponents(), mem, MaskMem(mem), sink=sink).run()
    assert [o.status for o in out] == [Status.OK, Status.OK]
    assert sink.saved["a"].labels.n_instances == 2
    assert out[0].report_row()["pieces"] == 2


def test_existing_outputs_are_skipped_and_failures_recorded():
    mask, _ = two_rings()
    mem = Memory({"a": (mask, None), "b": (mask, None)})
    sink = ListSink()
    sink.saved["a"] = None
    out = SegmentImages(Broken(), mem, MaskMem(mem), sink=sink).run()
    assert out[0].status is Status.SKIPPED
    assert out[1].status is Status.FAILED and "boom" in out[1].error


def test_evaluate_scores_perfect_method():
    mask, gt = two_rings()
    mem = Memory({"a": (mask, gt)})
    _, summary = EvaluateSegmenter(ConnectedComponents(), mem, MaskMem(mem), GtMem(mem)).run()
    assert summary.f1 == 1.0 and summary.n_images == 1


def test_match_instances_counts_split_off_piece_as_fp():
    _, gt = two_rings()
    pred = gt.copy()
    pred[:, 65:100][gt[:, 65:100] == 1] = 3  # cut ~1/4 off ring 1: the rest still matches (IoU > 0.5)
    m = match_instances(InstanceLabels(gt), InstanceLabels(pred))
    assert (m.tp, m.fp, m.fn) == (2, 1, 0)


def test_match_instances_ring_split_in_half_is_missed():
    _, gt = two_rings()
    pred = gt.copy()
    pred[:, 50:100][gt[:, 50:100] == 1] = 3  # halves: neither half reaches IoU 0.5 reliably
    m = match_instances(InstanceLabels(gt), InstanceLabels(pred), iou_threshold=0.6)
    assert (m.tp, m.fp, m.fn) == (1, 2, 1)


class ListSource(ImageSource):
    def __init__(self, refs):
        self.refs = refs

    def __iter__(self):
        return iter(self.refs)


class OneLabel(InstanceSegmenter):
    """Merges everything into one instance (wrong where rings are separate)."""
    name = "one"

    def segment(self, mask):
        return SegmentationResult(InstanceLabels(mask.data.astype(np.int32)), self.name)


def test_pair_auroc_perfect_and_merged():
    mask, gt = two_rings()
    pa, pb = sample_pixel_pairs(InstanceLabels(gt), n_pairs=5000, radius=120, seed=1)
    assert len(pa) == 5000
    perfect = pair_counts(InstanceLabels(gt), InstanceLabels(gt), pa, pb)
    merged = pair_counts(InstanceLabels(gt), InstanceLabels(mask.astype(np.int32)), pa, pb)
    assert perfect.auroc == 1.0
    assert merged.tn == 0 and merged.auroc == 0.5  # says "same" for every pair
    assert PairCounts(tp=3).auroc is None  # only one class present


def test_compare_runs_all_methods_on_same_pairs():
    mask, gt = two_rings()
    mem = Memory({"a": (mask, gt), "b": (mask, gt)})
    items, summary = CompareSegmenters({"cc": ConnectedComponents(), "one": OneLabel()}, mem, MaskMem(mem),
                                       GtMem(mem)).run()
    assert [set(it.outcomes) for it in items] == [{"cc", "one"}] * 2
    by = {m.method: m for m in summary.methods}
    assert by["cc"].evaluation.f1 == 1.0 and by["cc"].pair_auroc == 1.0
    assert by["one"].evaluation.f1 < 1.0 and by["one"].pair_auroc == 0.5
    a, b = items[0].outcomes["cc"].pairs, items[0].outcomes["one"].pairs
    assert a.tp + a.fn == b.tp + b.fn and a.fp + a.tn == b.fp + b.tn  # identical pairs


def test_compare_without_ground_truth_reports_timing_only():
    mask, _ = two_rings()
    mem = Memory({"a": (mask, None)})
    no_gt = ListSource([ImageRef("a", "a")])  # gt_path is None
    items, summary = CompareSegmenters({"cc": ConnectedComponents(), "one": OneLabel()}, no_gt,
                                       MaskMem(mem), GtMem(mem)).run()
    assert not items[0].has_gt
    assert all(m.evaluation is None and m.n_ok == 1 and m.total_seconds >= 0 for m in summary.methods)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
