"""Semantic segmentation: metrics, use case (with fakes), reference method, file adapters, CLI, plugins."""

import json
import os
import sys
import types
from importlib.metadata import EntryPoint

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memseg import registry, semantic  # noqa: E402
from memseg.application import SegmentMicrographs  # noqa: E402
from memseg.cli import main  # noqa: E402
from memseg.domain import (GroundTruthMaskReader, ImageRef, ImageSource, MembraneMask, Micrograph,  # noqa: E402
                           MicrographReader, PixelScores, ResultSink, SemanticResult, SemanticSegmenter,
                           SemanticSink, Status, pixel_scores, summarize_semantic)
from memseg.io import MicrographFileReader  # noqa: E402
from memseg.segmenters.components import ConnectedComponentsSegmenter  # noqa: E402
from memseg.semantic.base import SemanticSegmenterPlugin  # noqa: E402
from memseg.semantic.threshold import ThresholdConfig, ThresholdSegmenter  # noqa: E402


def rings_micrograph(seed=0, noise=0.3):
    """Two dark rings (membranes) on a bright, noisy, sloped background + the true mask."""
    yy, xx = np.mgrid[:240, :320]
    r1 = np.hypot(yy - 120, xx - 90)
    r2 = np.hypot(yy - 120, xx - 230)
    mask = (np.abs(r1 - 50) < 3) | (np.abs(r2 - 40) < 3)
    rng = np.random.default_rng(seed)
    img = 1.0 + 0.002 * xx - 1.0 * mask + noise * rng.standard_normal(mask.shape)
    return img.astype(np.float32), mask


# ───────────────────────── metrics ──

def test_pixel_scores_and_pooling():
    gt = np.zeros((4, 4), bool)
    gt[0, :] = True                     # 4 px
    pred = np.zeros((4, 4), bool)
    pred[0, :2] = True                  # 2 tp
    pred[3, 0] = True                   # 1 fp
    s = pixel_scores(MembraneMask(gt), MembraneMask(pred))
    assert (s.tp, s.fp, s.fn, s.tn) == (2, 1, 2, 11)
    assert s.iou == pytest.approx(2 / 5)
    assert s.dice == pytest.approx(4 / 7)
    pooled = summarize_semantic([s, PixelScores(tp=3)])
    assert pooled.pooled.tp == 5 and pooled.n_images == 2
    assert pooled.mean_iou == pytest.approx((0.4 + 1.0) / 2)


def test_pixel_scores_shape_mismatch():
    with pytest.raises(ValueError):
        pixel_scores(MembraneMask(np.zeros((2, 2))), MembraneMask(np.zeros((3, 3))))


# ───────────────────────── use case with fakes ──

class Images(ImageSource, MicrographReader, GroundTruthMaskReader):
    """In-memory micrographs and GT masks; doubles as source and readers."""

    def __init__(self, items):  # key -> (image, gt mask or None)
        self.items = items

    def __iter__(self):
        return iter(ImageRef(k, k, k if gt is not None else None) for k, (_, gt) in self.items.items())

    def read(self, ref):
        img, gt = self.items[ref.key]
        return Micrograph(img, ref, pixel_size=2.0)


class GT(GroundTruthMaskReader):
    def __init__(self, images):
        self.images = images

    def read(self, ref):
        return MembraneMask(self.images.items[ref.key][1], ref)


class Oracle(SemanticSegmenter):
    """Returns a stored mask; fails on request."""

    name = "oracle"

    def __init__(self, masks):
        self.masks = masks
        self.seen_pixel_sizes = []

    def segment(self, image):
        self.seen_pixel_sizes.append(image.pixel_size)
        m = self.masks[image.ref.key]
        if isinstance(m, Exception):
            raise m
        return SemanticResult(MembraneMask(m), self.name, probability=m.astype(np.float32), info=dict(x=1))


class Memo(SemanticSink, ResultSink):
    def __init__(self):
        self.saved = {}

    def save(self, ref, result):
        self.saved[ref.key] = result


def test_segment_micrographs_runs_semantic_then_instance_and_scores():
    img, mask = rings_micrograph()
    bad = np.zeros((5, 5), bool)
    images = Images({"a": (img, mask), "b": (img, None), "c": (img, mask), "d": (img, mask)})
    oracle = Oracle({"a": mask, "b": mask, "c": RuntimeError("boom"), "d": bad})
    sem_sink, inst_sink = Memo(), Memo()
    outcomes, summary = SegmentMicrographs(
        semantic=oracle, source=images, reader=images, instance=ConnectedComponentsSegmenter(min_size=10),
        semantic_sink=sem_sink, instance_sink=inst_sink, gt_reader=GT(images)).run()

    st = {o.ref.key: o.status for o in outcomes}
    assert st == {"a": Status.OK, "b": Status.OK, "c": Status.FAILED, "d": Status.FAILED}
    assert "boom" in outcomes[2].error and "mask for a" in outcomes[3].error   # wrong-shape mask rejected
    a, b = outcomes[0], outcomes[1]
    assert a.instances.labels.n_instances == 2                  # the two rings
    assert a.scores.iou == pytest.approx(1.0) and b.scores is None  # b has no GT
    assert summary.n_images == 1 and summary.pooled.iou == pytest.approx(1.0)
    assert set(sem_sink.saved) == {"a", "b"} and set(inst_sink.saved) == {"a", "b"}
    assert oracle.seen_pixel_sizes[0] == 2.0
    row = a.report_row()
    assert row["semantic_method"] == "oracle" and row["n_instances"] == 2 and row["iou"] == 1.0


def test_segment_micrographs_without_instance_or_gt():
    img, mask = rings_micrograph()
    images = Images({"a": (img, mask)})
    outcomes, summary = SegmentMicrographs(Oracle({"a": mask}), images, images).run()
    assert outcomes[0].status is Status.OK and outcomes[0].instances is None
    assert outcomes[0].scores is None and summary is None


# ───────────────────────── reference method ──

def test_threshold_finds_dark_rings():
    img, mask = rings_micrograph()
    res = ThresholdSegmenter(ThresholdConfig(k=4, smooth_sigma=1, min_area=50)).segment(Micrograph(img))
    s = pixel_scores(MembraneMask(mask), res.mask)
    assert s.iou > 0.85, s
    assert res.probability.shape == img.shape and 0 <= res.probability.min() <= res.probability.max() <= 1
    assert ConnectedComponentsSegmenter(min_size=50).segment(res.mask).labels.n_instances == 2


def test_threshold_rejects_unknown_parameter():
    with pytest.raises(ValueError, match="unknown threshold parameter"):
        ThresholdConfig.from_overrides({"nope": 1})


# ───────────────────────── file adapters + CLI ──

def test_mrc_reader_pixel_size(tmp_path):
    mrcfile = pytest.importorskip("mrcfile")
    img, _ = rings_micrograph()
    for apix, expected in ((1.7, 1.7), (1.0, 2.7)):   # 1.0 = "unset" -> default
        p = str(tmp_path / f"m_{apix}.mrc")
        with mrcfile.new(p) as m:
            m.set_data(img)
            m.voxel_size = apix
        mg = MicrographFileReader(default_pixel_size=2.7).read(ImageRef(p, "m.mrc"))
        assert mg.pixel_size == pytest.approx(expected, rel=1e-5)
        assert np.allclose(mg.data, img)


def test_cli_semantic_end_to_end(tmp_path, capsys):
    import tifffile
    img, mask = rings_micrograph()
    (tmp_path / "in").mkdir()
    (tmp_path / "gt").mkdir()
    tifffile.imwrite(tmp_path / "in" / "mic.tif", img)
    tifffile.imwrite(tmp_path / "gt" / "mic.tif", mask.astype(np.uint8))
    out = tmp_path / "out"
    rc = main(["semantic", str(tmp_path / "in"), "-o", str(out), "--set", "k=4", "--set", "smooth_sigma=1", "--set", "min_area=50",
               "--instance", "components", "--instance_set", "min_size=50", "--gt_dir", str(tmp_path / "gt"),
               "--save_prob", "--preview"])
    assert rc == 0
    for f in ("mic_mask.tif", "mic_prob.tif", "mic_mask.png", "instances/mic.tif", "instances/mic_preview.png",
              "summary.csv", "summary.json"):
        assert (out / f).is_file(), f
    assert tifffile.imread(out / "instances" / "mic.tif").max() == 2
    assert json.loads((out / "summary.json").read_text())["iou"] > 0.85
    assert "IoU" in capsys.readouterr().out
    # second run skips: mask exists
    assert main(["semantic", str(tmp_path / "in"), "-o", str(out)]) == 0
    assert "skipped" in capsys.readouterr().out


def test_cli_semantic_unknown_methods(capsys):
    assert main(["semantic", "x", "-o", "y", "--method", "nope"]) == 2
    assert main(["semantic", "x", "-o", "y", "--instance", "nope"]) == 2


# ───────────────────────── plugins ──

class _FakeSemanticPlugin(SemanticSegmenterPlugin):
    name = "fakesem"
    summary = "fake semantic method"

    def build(self, args, overrides):
        return ThresholdSegmenter()


def test_semantic_entry_points_are_discovered(monkeypatch, capsys):
    mod = types.ModuleType("fake_memseg_semantic")
    mod.Plugin = _FakeSemanticPlugin
    monkeypatch.setitem(sys.modules, "fake_memseg_semantic", mod)
    eps = [EntryPoint(name="fakesem", value="fake_memseg_semantic:Plugin", group=semantic.ENTRY_POINT_GROUP)]
    monkeypatch.setattr(registry, "entry_points", lambda group: eps if group == semantic.ENTRY_POINT_GROUP else [])
    assert semantic.available() == ["threshold", "fakesem"]
    assert main(["methods"]) == 0
    out = capsys.readouterr().out
    assert "fake semantic method" in out and "semantic methods" in out
