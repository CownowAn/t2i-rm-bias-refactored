from __future__ import annotations
import asyncio
from typing import Any, Coroutine, TypeVar
from tqdm.asyncio import tqdm_asyncio

T = TypeVar("T")


async def bounded_gather(
    coros: list[Coroutine[Any, Any, T]],
    max_parallel: int,
    desc: str = "",
) -> list[T]:
    """Gather coroutines with a concurrency limit and optional tqdm progress bar."""
    if not coros:
        return []

    sem = asyncio.Semaphore(max_parallel)

    async def _wrap(coro: Coroutine[Any, Any, T]) -> T:
        async with sem:
            return await coro

    wrapped = [_wrap(c) for c in coros]
    if desc:
        return await tqdm_asyncio.gather(*wrapped, desc=desc)
    return await asyncio.gather(*wrapped)


class GpuApplierPool:
    """asyncio.Queue-based pool of FluxKontextApplier instances (one per GPU).

    GPU inference (run_in_executor) is serialized per device via this pool.
    Multiple coroutines can wait for a free slot concurrently.
    """

    def __init__(self, appliers: list) -> None:
        self._appliers = appliers
        self._queue: asyncio.Queue | None = None
        self._init_lock: asyncio.Lock | None = None

    async def _get_queue(self) -> asyncio.Queue:
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._queue is None:
                q: asyncio.Queue = asyncio.Queue()
                for a in self._appliers:
                    await q.put(a)
                self._queue = q
        return self._queue

    async def apply(self, image_path: str, instruction: str, output_path: str) -> str:
        """Acquire a free GPU applier, run inference, release it."""
        queue = await self._get_queue()
        applier = await queue.get()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, applier.apply, image_path, instruction, output_path
            )
            return result
        finally:
            await queue.put(applier)
