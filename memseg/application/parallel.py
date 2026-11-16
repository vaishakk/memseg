"""Per-image work of the image use cases, run sequentially or through an injected Executor.

Without an executor every image runs `segmenter.segment` in this process, one after another.
With one (e.g. a ProcessPoolExecutor created by the composition root):

* a plain `InstanceSegmenter` runs read -> segment -> save (-> match) as one job per image;
* a `StagedSegmenter` runs `prepare` and `finish` (+ read / save / match) as jobs, and `infer`
  in this process on batches of prepared images, so a GPU model is loaded once and fed in batches
  while post-processing runs for several images at once.

Everything sent to a job (reader, sink, ground-truth reader, stages, plain segmenter) must be
picklable when the executor uses processes. The job functions are module-level for that reason.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
from dataclasses import dataclass
from typing import Any

from ..domain.metrics import match_instances
from ..domain.models import ImageRef, InstanceLabels, ItemOutcome, MembraneMask, SegmentationResult, Status
from ..domain.ports import (
    GroundTruthReader,
    InstanceSegmenter,
    MaskReader,
    ResultSink,
    SegmenterStages,
    StagedSegmenter,
)


def describe_error(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


@dataclass(frozen=True)
class ImageJob:
    """The per-image I/O of a use case: read the mask (and ground truth), save the result, score it."""

    reader: MaskReader
    sink: ResultSink
    gt_reader: GroundTruthReader | None = None  # set: read ground truth and match against it
    iou_threshold: float = 0.5

    def read(self, ref: ImageRef) -> tuple[MembraneMask, InstanceLabels | None]:
        if self.gt_reader is not None and ref.gt_path is None:
            raise ValueError("no ground truth for this image")
        mask = self.reader.read(ref)
        gt = self.gt_reader.read(ref) if self.gt_reader is not None else None
        return mask, gt

    def complete(
        self,
        ref: ImageRef,
        mask: MembraneMask,
        gt: InstanceLabels | None,
        result: SegmentationResult,
        t0: float,
        extra_seconds: float = 0.0,
    ) -> ItemOutcome:
        self.sink.save(ref, result)
        m = match_instances(gt, result.labels, self.iou_threshold) if gt is not None else None
        return ItemOutcome(
            ref=ref,
            status=Status.OK,
            seconds=time.time() - t0 + extra_seconds,
            shape=mask.shape,
            result=result,
            match=m,
        )


def _failed(ref: ImageRef, e: BaseException, seconds: float = 0.0) -> ItemOutcome:
    return ItemOutcome(ref=ref, status=Status.FAILED, seconds=seconds, error=describe_error(e))


# ───────────────────────────────────────────── jobs (module-level: picklable) ──


def whole_job(job: ImageJob, segmenter: InstanceSegmenter, ref: ImageRef) -> ItemOutcome:
    """read -> segment -> save (-> match) for one image; failures become a FAILED outcome."""
    t0 = time.time()
    try:
        mask, gt = job.read(ref)
        return job.complete(ref, mask, gt, segmenter.segment(mask), t0)
    except Exception as e:  # keep going on bad inputs
        return _failed(ref, e, time.time() - t0)


@dataclass
class _Prepared:
    mask: MembraneMask
    gt: InstanceLabels | None
    prepared: Any
    seconds: float


def head_job(job: ImageJob, stages: SegmenterStages, ref: ImageRef) -> _Prepared:
    t0 = time.time()
    mask, gt = job.read(ref)
    prepared = stages.prepare(mask)
    return _Prepared(mask, gt, prepared, time.time() - t0)


def tail_job(
    job: ImageJob, stages: SegmenterStages, ref: ImageRef, head: _Prepared, inferred: Any, seconds: float
) -> ItemOutcome:
    t0 = time.time()
    try:
        result = stages.finish(head.mask, head.prepared, inferred)
        return job.complete(ref, head.mask, head.gt, result, t0, extra_seconds=seconds)
    except Exception as e:
        return _failed(ref, e, time.time() - t0 + seconds)


# ───────────────────────────────────────────── runner ──


def run_images(
    refs: Sequence[ImageRef],
    segmenter: InstanceSegmenter,
    job: ImageJob,
    on_done: Callable[[ItemOutcome], None],
    *,
    skip: Callable[[ImageRef], bool] | None = None,
    executor: Executor | None = None,
    batch_size: int | None = None,
    max_in_flight: int = 8,
) -> list[ItemOutcome]:
    """Process `refs`; returns one outcome per ref, in input order.

    `on_done` is called once per image as it finishes (input order when sequential,
    completion order otherwise). `skip(ref)` -> True marks an image SKIPPED without work.
    `max_in_flight` bounds the images read but not yet finished (memory); `batch_size`
    overrides the staged segmenter's preferred `infer` batch.
    """
    outcomes: dict[int, ItemOutcome] = {}

    def done(i: int, o: ItemOutcome) -> None:
        outcomes[i] = o
        on_done(o)

    if executor is None:
        for i, ref in enumerate(refs):
            if skip is not None and skip(ref):
                done(i, ItemOutcome(ref=ref, status=Status.SKIPPED))
            else:
                done(i, whole_job(job, segmenter, ref))
        return [outcomes[i] for i in range(len(refs))]

    todo: deque[int] = deque()
    for i, ref in enumerate(refs):
        if skip is not None and skip(ref):
            done(i, ItemOutcome(ref=ref, status=Status.SKIPPED))
        else:
            todo.append(i)

    if isinstance(segmenter, StagedSegmenter):
        bs = max(1, batch_size if batch_size is not None else segmenter.batch_size)
        _run_staged(refs, segmenter, job, executor, todo, done, bs, max(max_in_flight, bs))
    else:
        _run_plain(refs, segmenter, job, executor, todo, done, max(max_in_flight, 1))
    return [outcomes[i] for i in range(len(refs))]


def _submit(executor: Executor, fn: Callable[..., Any], *args: Any) -> Future:
    """executor.submit; a refusal (e.g. a broken process pool) becomes a failed future."""
    try:
        return executor.submit(fn, *args)
    except Exception as e:
        f: Future = Future()
        f.set_exception(e)
        return f


def _run_plain(
    refs: Sequence[ImageRef],
    segmenter: InstanceSegmenter,
    job: ImageJob,
    executor: Executor,
    todo: deque[int],
    done: Callable[[int, ItemOutcome], None],
    max_in_flight: int,
) -> None:
    running: dict[Future, int] = {}
    while todo or running:
        while todo and len(running) < max_in_flight:
            i = todo.popleft()
            running[_submit(executor, whole_job, job, segmenter, refs[i])] = i
        finished, _ = wait(running, return_when=FIRST_COMPLETED)
        for f in finished:
            i = running.pop(f)
            err = f.exception()
            done(i, _failed(refs[i], err) if err is not None else f.result())


def _run_staged(
    refs: Sequence[ImageRef],
    segmenter: StagedSegmenter,
    job: ImageJob,
    executor: Executor,
    todo: deque[int],
    done: Callable[[int, ItemOutcome], None],
    batch_size: int,
    max_in_flight: int,
) -> None:
    stages = segmenter.stages()
    heads: dict[Future, int] = {}  # prepare running
    ready: list[tuple[int, _Prepared]] = []  # prepared, waiting for infer
    tails: dict[Future, int] = {}  # finish running
    in_flight = 0

    while todo or heads or ready or tails:
        while todo and in_flight < max_in_flight:
            i = todo.popleft()
            heads[_submit(executor, head_job, job, stages, refs[i])] = i
            in_flight += 1

        # infer when a batch is full, or when no more prepared images are on their way
        if ready and (len(ready) >= batch_size or not heads):
            batch, ready = ready[:batch_size], ready[batch_size:]
            t0 = time.time()
            try:
                inferred = segmenter.infer([p.prepared for _, p in batch])
                if len(inferred) != len(batch):
                    raise ValueError(f"infer returned {len(inferred)} results for {len(batch)} inputs")
            except Exception as e:
                share = (time.time() - t0) / len(batch)
                for i, p in batch:
                    done(i, _failed(refs[i], e, p.seconds + share))
                    in_flight -= 1
                continue
            share = (time.time() - t0) / len(batch)
            for (i, p), out in zip(batch, inferred, strict=True):
                tails[_submit(executor, tail_job, job, stages, refs[i], p, out, p.seconds + share)] = i
            continue

        finished, _ = wait(list(heads) + list(tails), return_when=FIRST_COMPLETED)
        for f in finished:
            err = f.exception()
            if f in heads:
                i = heads.pop(f)
                if err is not None:
                    done(i, _failed(refs[i], err))
                    in_flight -= 1
                else:
                    ready.append((i, f.result()))
            else:
                i = tails.pop(f)
                done(i, _failed(refs[i], err) if err is not None else f.result())
                in_flight -= 1
