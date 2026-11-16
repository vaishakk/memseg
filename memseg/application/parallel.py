"""Per-image work of the use cases, run sequentially or through an injected Executor.

A *job* describes the per-image I/O of a use case: read the input, and, given the method's
result, save it / run follow-up steps / score it (`ImageJob` for instance masks,
`MicrographJob` for semantic segmentation). The *method* is either plain (`segment(x)`) or staged
(`stages().prepare` -> `infer` on batches -> `stages().finish`).

Without an executor every image runs `segment` in this process, one after another.
With one (e.g. a ProcessPoolExecutor created by the composition root):

* a plain method runs read -> segment -> complete as one job per image;
* a staged method runs read -> prepare and finish -> complete as jobs, and `infer` in this
  process on batches of prepared images, so a GPU model is loaded once and fed in batches while
  the CPU work runs for several images at once.

Everything sent to a job (the job object with its readers / sinks / follow-up method, the stages,
a plain method) must be picklable when the executor uses processes. The job functions are
module-level for that reason.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
from dataclasses import dataclass, field
from typing import Any

from ..domain.metrics import match_instances, pixel_scores
from ..domain.models import (
    ImageRef,
    InstanceLabels,
    ItemOutcome,
    MembraneMask,
    Micrograph,
    MicrographOutcome,
    SegmentationResult,
    SemanticResult,
    Status,
)
from ..domain.ports import (
    GroundTruthMaskReader,
    GroundTruthReader,
    InstanceSegmenter,
    MaskReader,
    MicrographReader,
    NullSink,
    ResultSink,
    SemanticSink,
    StagedSegmenter,
    StagedSemanticSegmenter,
)


def describe_error(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


# ───────────────────────────────────────────── jobs: per-image I/O of a use case ──


@dataclass(frozen=True)
class ImageJob:
    """Instance segmentation of masks: read the mask (and ground truth), save the result, score it."""

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

    @staticmethod
    def method_input(data: tuple[MembraneMask, InstanceLabels | None]) -> MembraneMask:
        return data[0]

    def complete(
        self,
        ref: ImageRef,
        data: tuple[MembraneMask, InstanceLabels | None],
        result: SegmentationResult,
        seconds: float,
    ) -> ItemOutcome:
        t0 = time.time()
        mask, gt = data
        self.sink.save(ref, result)
        m = match_instances(gt, result.labels, self.iou_threshold) if gt is not None else None
        return ItemOutcome(
            ref=ref,
            status=Status.OK,
            seconds=seconds + time.time() - t0,
            shape=mask.shape,
            result=result,
            match=m,
        )

    @staticmethod
    def failed(ref: ImageRef, e: BaseException, seconds: float = 0.0) -> ItemOutcome:
        return ItemOutcome(ref=ref, status=Status.FAILED, seconds=seconds, error=describe_error(e))

    @staticmethod
    def skipped(ref: ImageRef) -> ItemOutcome:
        return ItemOutcome(ref=ref, status=Status.SKIPPED)


@dataclass(frozen=True)
class MicrographJob:
    """Semantic segmentation of micrographs: read, then (after the semantic method) check and save
    the mask, run the optional instance method on it and save that, and score against a GT mask."""

    reader: MicrographReader
    semantic_sink: SemanticSink
    instance: InstanceSegmenter | None = None
    instance_sink: ResultSink = field(default_factory=NullSink)
    gt_reader: GroundTruthMaskReader | None = None

    def read(self, ref: ImageRef) -> Micrograph:
        return self.reader.read(ref)

    @staticmethod
    def method_input(data: Micrograph) -> Micrograph:
        return data

    def complete(self, ref: ImageRef, image: Micrograph, sem: SemanticResult, seconds: float) -> MicrographOutcome:
        t0 = time.time()
        if sem.mask.shape != image.shape:
            raise ValueError(f"method returned a {sem.mask.shape} mask for a {image.shape} image")
        sem.mask.ref = ref
        self.semantic_sink.save(ref, sem)
        inst = None
        if self.instance is not None:
            inst = self.instance.segment(sem.mask)
            self.instance_sink.save(ref, inst)
        scores = None
        if self.gt_reader is not None and ref.gt_path is not None:
            scores = pixel_scores(self.gt_reader.read(ref), sem.mask)
        return MicrographOutcome(
            ref=ref,
            status=Status.OK,
            seconds=seconds + time.time() - t0,
            shape=image.shape,
            semantic=sem,
            instances=inst,
            scores=scores,
        )

    @staticmethod
    def failed(ref: ImageRef, e: BaseException, seconds: float = 0.0) -> MicrographOutcome:
        return MicrographOutcome(ref=ref, status=Status.FAILED, seconds=seconds, error=describe_error(e))

    @staticmethod
    def skipped(ref: ImageRef) -> MicrographOutcome:
        return MicrographOutcome(ref=ref, status=Status.SKIPPED)


Job = ImageJob | MicrographJob
Staged = StagedSegmenter | StagedSemanticSegmenter


# ───────────────────────────────────────────── worker functions (module-level: picklable) ──


def whole_job(job: Job, method: Any, ref: ImageRef) -> Any:
    """read -> segment -> complete for one image; failures become a FAILED outcome."""
    t0 = time.time()
    try:
        data = job.read(ref)
        result = method.segment(job.method_input(data))  # type: ignore[arg-type]  # data matches its own job type
        return job.complete(ref, data, result, time.time() - t0)  # type: ignore[arg-type]  # data matches its own job type
    except Exception as e:  # keep going on bad inputs
        return job.failed(ref, e, time.time() - t0)


@dataclass
class _Prepared:
    data: Any
    prepared: Any
    seconds: float


def head_job(job: Job, stages: Any, ref: ImageRef) -> _Prepared:
    t0 = time.time()
    data = job.read(ref)
    prepared = stages.prepare(job.method_input(data))  # type: ignore[arg-type]  # data matches its own job type
    return _Prepared(data, prepared, time.time() - t0)


def tail_job(job: Job, stages: Any, ref: ImageRef, head: _Prepared, inferred: Any, seconds: float) -> Any:
    t0 = time.time()
    try:
        result = stages.finish(job.method_input(head.data), head.prepared, inferred)
        return job.complete(ref, head.data, result, seconds + time.time() - t0)
    except Exception as e:
        return job.failed(ref, e, seconds + time.time() - t0)


# ───────────────────────────────────────────── runner ──


def run_images(
    refs: Sequence[ImageRef],
    method: Any,
    job: Job,
    on_done: Callable[[Any], None],
    *,
    skip: Callable[[ImageRef], bool] | None = None,
    executor: Executor | None = None,
    batch_size: int | None = None,
    max_in_flight: int = 8,
) -> list[Any]:
    """Process `refs` with `method` (instance or semantic, plain or staged, matching `job`).

    Returns one outcome per ref, in input order. `on_done` is called once per image as it finishes
    (input order when sequential, completion order otherwise). `skip(ref)` -> True marks an image
    SKIPPED without work. `max_in_flight` bounds the images read but not yet finished (memory);
    `batch_size` overrides a staged method's preferred `infer` batch.
    """
    outcomes: dict[int, Any] = {}

    def done(i: int, o: Any) -> None:
        outcomes[i] = o
        on_done(o)

    if executor is None:
        for i, ref in enumerate(refs):
            if skip is not None and skip(ref):
                done(i, job.skipped(ref))
            else:
                done(i, whole_job(job, method, ref))
        return [outcomes[i] for i in range(len(refs))]

    todo: deque[int] = deque()
    for i, ref in enumerate(refs):
        if skip is not None and skip(ref):
            done(i, job.skipped(ref))
        else:
            todo.append(i)

    if isinstance(method, StagedSegmenter | StagedSemanticSegmenter):
        bs = max(1, batch_size if batch_size is not None else method.batch_size)
        _run_staged(refs, method, job, executor, todo, done, bs, max(max_in_flight, bs))
    else:
        _run_plain(refs, method, job, executor, todo, done, max(max_in_flight, 1))
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
    method: Any,
    job: Job,
    executor: Executor,
    todo: deque[int],
    done: Callable[[int, Any], None],
    max_in_flight: int,
) -> None:
    running: dict[Future, int] = {}
    while todo or running:
        while todo and len(running) < max_in_flight:
            i = todo.popleft()
            running[_submit(executor, whole_job, job, method, refs[i])] = i
        finished, _ = wait(running, return_when=FIRST_COMPLETED)
        for f in finished:
            i = running.pop(f)
            err = f.exception()
            done(i, job.failed(refs[i], err) if err is not None else f.result())


def _run_staged(
    refs: Sequence[ImageRef],
    method: Staged,
    job: Job,
    executor: Executor,
    todo: deque[int],
    done: Callable[[int, Any], None],
    batch_size: int,
    max_in_flight: int,
) -> None:
    stages = method.stages()
    heads: dict[Future, int] = {}  # read + prepare running
    ready: list[tuple[int, _Prepared]] = []  # prepared, waiting for infer
    tails: dict[Future, int] = {}  # finish + complete running
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
                inferred = method.infer([p.prepared for _, p in batch])
                if len(inferred) != len(batch):
                    raise ValueError(f"infer returned {len(inferred)} results for {len(batch)} inputs")
            except Exception as e:
                share = (time.time() - t0) / len(batch)
                for i, p in batch:
                    done(i, job.failed(refs[i], e, p.seconds + share))
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
                    done(i, job.failed(refs[i], err))
                    in_flight -= 1
                else:
                    ready.append((i, f.result()))
            else:
                i = tails.pop(f)
                done(i, job.failed(refs[i], err) if err is not None else f.result())
                in_flight -= 1
