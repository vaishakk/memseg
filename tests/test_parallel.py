"""Parallel execution of the image use cases (staged and plain segmenters, real worker processes).

The fakes are module-level so that spawned worker processes can unpickle them.
"""

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memseg.application import EvaluateSegmenter, SegmentImages
from memseg.cli import main
from memseg.domain import (
    GroundTruthReader,
    ImageRef,
    ImageSource,
    MaskReader,
    MembraneMask,
    SegmenterStages,
    StagedSegmenter,
    Status,
)
from memseg.io import LabelTiffSink
from memseg.segmenters.components import ConnectedComponentsSegmenter

BAD_SHAPES = {"badprep": (7, 7), "badinfer": (9, 9), "badfinish": (11, 11)}


def synthetic_mask(key: str) -> np.ndarray:
    """A few separate blobs, different per key; special keys give shapes that make a stage fail."""
    if key in BAD_SHAPES:
        return np.ones(BAD_SHAPES[key], bool)
    rng = np.random.default_rng(sum(map(ord, key)))
    m = np.zeros((64, 80), bool)
    for _ in range(rng.integers(1, 5)):
        y, x = rng.integers(2, 50), rng.integers(2, 66)
        m[y : y + 10, x : x + 12] = True
    return m


class Source(ImageSource):
    def __init__(self, keys):
        self.keys = keys

    def __iter__(self):
        return iter(ImageRef(k, k, k) for k in self.keys)


class Reader(MaskReader):
    def read(self, ref):
        if ref.key == "badread":
            raise OSError("cannot read badread")
        return MembraneMask(synthetic_mask(ref.key), ref)


class GT(GroundTruthReader):
    def read(self, ref):
        return ConnectedComponentsSegmenter(min_size=1).segment(MembraneMask(synthetic_mask(ref.key))).labels


class Stages(SegmenterStages):
    def prepare(self, mask):
        if mask.shape == BAD_SHAPES["badprep"]:
            raise ValueError("prepare failed")
        return mask.data.astype(np.uint8)

    def finish(self, mask, prepared, inferred):
        if mask.shape == BAD_SHAPES["badfinish"]:
            raise ValueError("finish failed")
        res = ConnectedComponentsSegmenter(min_size=1).segment(MembraneMask(inferred > 0))
        res.info["pid"] = os.getpid()
        return res


class Staged(StagedSegmenter):
    """Inference = identity; records the batches it was given (in the main process)."""

    name = "staged"

    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.batches = []

    def stages(self):
        return Stages()

    def infer(self, prepared):
        self.batches.append(len(prepared))
        if any(p.shape == BAD_SHAPES["badinfer"] for p in prepared):
            raise RuntimeError("infer failed")
        return [p.copy() for p in prepared]


@pytest.fixture(scope="module")
def pool():
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as ex:
        yield ex


KEYS = ["a", "b", "badread", "c", "badprep", "d", "badfinish", "e", "f", "g"]


def _summary(outcomes):
    return [
        (o.ref.key, o.status, None if o.result is None else o.result.labels.data.tobytes(), o.error and o.error[:20])
        for o in outcomes
    ]


def test_staged_segment_is_prepare_infer_finish():
    mask = MembraneMask(synthetic_mask("a"))
    direct = ConnectedComponentsSegmenter(min_size=1).segment(mask).labels.data
    assert np.array_equal(Staged().segment(mask).labels.data, direct)


def test_parallel_staged_matches_sequential_and_keeps_order(pool, tmp_path):
    seq = SegmentImages(Staged(), Source(KEYS), Reader(), LabelTiffSink(str(tmp_path / "s"))).run()
    seg = Staged(batch_size=3)
    par = SegmentImages(
        seg, Source(KEYS), Reader(), LabelTiffSink(str(tmp_path / "p")), executor=pool, max_in_flight=4
    ).run()
    assert [o.ref.key for o in par] == KEYS
    assert _summary(par) == _summary(seq)
    by_key = {o.ref.key: o for o in par}
    assert "cannot read" in by_key["badread"].error
    assert "prepare failed" in by_key["badprep"].error
    assert "finish failed" in by_key["badfinish"].error
    ok = [o for o in par if o.status is Status.OK]
    assert {o.result.info["pid"] for o in ok} - {os.getpid()}, "finish should run in worker processes"
    assert all(1 <= b <= 3 for b in seg.batches) and sum(seg.batches) == len(KEYS) - 2  # badread, badprep
    for k in ("a", "e", "g"):
        assert (tmp_path / "p" / f"{k}.tif").read_bytes() == (tmp_path / "s" / f"{k}.tif").read_bytes()


def test_infer_failure_fails_its_batch_only(tmp_path):
    keys = ["a", "badinfer", "b"]
    with ThreadPoolExecutor(2) as ex:
        out = SegmentImages(Staged(batch_size=1), Source(keys), Reader(), executor=ex).run()
    st = {o.ref.key: o.status for o in out}
    assert st == {"a": Status.OK, "badinfer": Status.FAILED, "b": Status.OK}
    assert "infer failed" in out[1].error


def test_parallel_skips_existing(pool, tmp_path):
    sink = LabelTiffSink(str(tmp_path))
    SegmentImages(Staged(), Source(["a"]), Reader(), sink).run()
    out = SegmentImages(Staged(), Source(["a", "b"]), Reader(), sink, executor=pool).run()
    assert [o.status for o in out] == [Status.SKIPPED, Status.OK]


def test_parallel_plain_segmenter_matches_sequential(pool):
    keys = ["a", "badread", "b", "c"]
    seg = ConnectedComponentsSegmenter(min_size=1)
    seq = SegmentImages(seg, Source(keys), Reader()).run()
    par = SegmentImages(seg, Source(keys), Reader(), executor=pool).run()
    assert _summary(par) == _summary(seq)


def test_parallel_evaluate_matches_sequential(pool):
    keys = ["a", "b", "badread", "c"]
    seq, s1 = EvaluateSegmenter(Staged(), Source(keys), Reader(), GT()).run()
    par, s2 = EvaluateSegmenter(Staged(batch_size=2), Source(keys), Reader(), GT(), executor=pool).run()
    assert _summary(par) == _summary(seq)
    assert s1 == s2 and s2.f1 == pytest.approx(1.0)


def test_cli_workers(tmp_path, capsys):
    import tifffile

    (tmp_path / "in").mkdir()
    for k in ("a", "b", "c"):
        tifffile.imwrite(tmp_path / "in" / f"{k}.tif", synthetic_mask(k).astype(np.uint8) * 255)
    base = ["segment", str(tmp_path / "in"), "--set", "min_size=1", "-o"]
    assert main([*base, str(tmp_path / "o1")]) == 0
    assert main([*base, str(tmp_path / "o2"), "--workers", "2"]) == 0
    for k in ("a", "b", "c"):
        assert (tmp_path / "o1" / f"{k}.tif").read_bytes() == (tmp_path / "o2" / f"{k}.tif").read_bytes()
    assert "3 ok" in capsys.readouterr().out
    assert main(["segment", str(tmp_path / "in"), "-o", str(tmp_path / "o3"), "--batch_size", "0"]) == 2
