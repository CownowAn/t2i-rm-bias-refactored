"""Multi-GPU + concurrency-safe wrapper around FluxKontextApplier.

Single process, N GPUs:
    devices = ["cuda:0", "cuda:1", ...]
    EditorRunner builds N FluxKontextApplier instances and serves them through
    `search.utils.async_utils.GpuApplierPool` — coroutines queue for a free
    GPU and run their FLUX inference in a thread executor.

Cross-process safety:
    The sidecar O_EXCL lockfile + per-PID tmp + `os.replace` rename pattern
    (originally proven in `baselines/score_baselines.py`) makes the cache at
    `<cf_root>/edits/...` safe to share across runs and machines.
"""
from __future__ import annotations

import datetime as _dt
import errno
import os
import time
from pathlib import Path

from loguru import logger

from debias.counterfactual.schemas import EditTask
from search.models.editor.flux_kontext import FluxKontextApplier
from search.utils.async_utils import GpuApplierPool, bounded_gather
from search.utils.io import save_json


METADATA_SCHEMA_VERSION = 1


# ── Cross-process lock (POSIX + NFS-friendly) ─────────────────────────────────


class _SideCarLock:
    """O_CREAT|O_EXCL on `<target>.lock`. Stale-recovers via PID liveness."""

    def __init__(self, target: Path, *, timeout: int = 1800, stale_after: int = 1800):
        self.target = Path(target)
        self.lock_path = self.target.with_suffix(self.target.suffix + ".lock")
        self.timeout = timeout
        self.stale_after = stale_after
        self._fd: int | None = None

    def _maybe_clear_stale(self) -> None:
        try:
            age = time.time() - self.lock_path.stat().st_mtime
        except FileNotFoundError:
            return
        if age < self.stale_after:
            return
        try:
            pid = int(self.lock_path.read_text().strip().splitlines()[0])
        except (OSError, ValueError, IndexError):
            pid = -1
        if pid > 0:
            try:
                os.kill(pid, 0)
                return                     # owner still alive
            except ProcessLookupError:
                pass
            except PermissionError:
                return                     # alive in a different uid namespace
        logger.warning(f"removing stale lock {self.lock_path} (age={age:.0f}s)")
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self) -> "_SideCarLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.timeout
        waited = False
        while True:
            try:
                self._fd = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
                os.write(self._fd, f"{os.getpid()}\n".encode())
                os.fsync(self._fd)
                return self
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"could not acquire {self.lock_path} within {self.timeout}s"
                    ) from None
                if not waited:
                    logger.debug(f"waiting for lock {self.lock_path}")
                    waited = True
                self._maybe_clear_stale()
                time.sleep(2.0)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    continue
                raise

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass


# ── Main runner ───────────────────────────────────────────────────────────────


class EditorRunner:
    """One process, N GPU appliers, asyncio.Queue-based pool. Idempotent."""

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-Kontext-dev",
        devices: list[str] | None = None,
        guidance_scale: float = 2.5,
        hf_cache_dir: str | None = None,
        max_parallel: int | None = None,
        *,
        lock_timeout: int = 1800,
        stale_after: int = 1800,
    ):
        devices = list(devices) if devices else ["cuda:0"]
        self._model_name = model_name
        self._guidance_scale = float(guidance_scale)
        self._devices = devices
        self._appliers = [
            FluxKontextApplier(
                model_name=model_name,
                device=d,
                guidance_scale=guidance_scale,
                hf_cache_dir=hf_cache_dir,
            )
            for d in devices
        ]
        self._pool = GpuApplierPool(self._appliers)
        self._max_parallel = max_parallel or len(devices)
        self._lock_timeout = lock_timeout
        self._stale_after = stale_after
        logger.info(
            f"EditorRunner: {len(devices)} GPU(s) → {devices}, "
            f"max_parallel={self._max_parallel}"
        )

    # ── Metadata ──────────────────────────────────────────────────────────

    def _write_metadata_sidecar(self, task: EditTask) -> None:
        """Write `<image_id>.json` next to the PNG with editor model + prompts."""
        out = Path(task.edited_output_path)
        meta_path = out.with_suffix(".json")
        meta = {
            "schema_version": METADATA_SCHEMA_VERSION,
            "editor_model": self._model_name,
            "guidance_scale": self._guidance_scale,
            "devices": self._devices,              # pool of devices used by this run
            "instruction": task.instruction,       # FLUX-Kontext edit command
            "source_prompt": task.source.prompt_text,  # original T2I prompt
            "topic_id": task.selection.topic_id,
            "attr": task.selection.attr,
            "source_image_id": task.source.image_id,
            "source_image_path": str(task.source.image_path),
            "created_at": _dt.datetime.now(_dt.timezone.utc)
                            .isoformat(timespec="seconds")
                            .replace("+00:00", "Z"),
        }
        try:
            save_json(meta, meta_path)
        except Exception as e:
            logger.warning(f"  could not write metadata sidecar at {meta_path}: {e}")

    # ── Public async API ──────────────────────────────────────────────────

    async def edit_one(self, task: EditTask) -> Path:
        """Concurrency-safe edit: per-output lock + atomic rename + sidecar.

        Fast path:   output already exists ⇒ return (backfill metadata if missing).
        Slow path:   acquire `<out>.lock`, re-check, await `pool.apply` (FLUX
                     runs in a thread executor on the next free GPU), rename,
                     write metadata sidecar.
        """
        out = Path(task.edited_output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            if not out.with_suffix(".json").exists():
                self._write_metadata_sidecar(task)
            return out

        with _SideCarLock(out, timeout=self._lock_timeout, stale_after=self._stale_after):
            if out.exists():                       # another process finished while we waited
                if not out.with_suffix(".json").exists():
                    self._write_metadata_sidecar(task)
                return out
            # Keep the original suffix at the END so PIL.Image.save can infer the
            # format from it. `<image_id>.png.tmp.<PID>` would make PIL see
            # `.<PID>` and fail with "unknown file extension".
            tmp = out.with_name(f"{out.stem}.tmp.{os.getpid()}{out.suffix}")
            try:
                await self._pool.apply(
                    image_path=str(task.source.image_path),
                    instruction=task.instruction,
                    output_path=str(tmp),
                )
                os.replace(tmp, out)               # atomic rename within same dir
                self._write_metadata_sidecar(task)
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
        return out

    async def edit_many(self, tasks: list[EditTask]) -> list[Path]:
        """Run all tasks concurrently up to `max_parallel` (= #devices by default)."""
        if not tasks:
            return []
        t0 = time.monotonic()
        coros = [self.edit_one(t) for t in tasks]
        outs = await bounded_gather(
            coros,
            max_parallel=self._max_parallel,
            desc=f"editing ({len(tasks)} tasks, {len(self._devices)} GPU)",
        )
        elapsed = time.monotonic() - t0
        logger.info(
            f"  edited {len(tasks)} tasks in {elapsed:.1f}s "
            f"({elapsed / max(len(tasks), 1):.2f}s/edit avg across {len(self._devices)} GPU(s))"
        )
        return outs