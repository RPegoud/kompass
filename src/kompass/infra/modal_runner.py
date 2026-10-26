import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Literal

import modal
from omegaconf import OmegaConf

from ..harness.schema import BenchResult, Payload


@dataclass(frozen=True)
class ModalConfig:
    cuda_volume_name: str
    triton_volume_name: str
    kernel_remote_path: str
    cuda_kernel_cache_path: str
    triton_kernel_cache_path: str
    max_containers: int
    timeout: int

    @classmethod
    def from_yaml(cls, path: Path):
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**raw)  # type: ignore


cfg = ModalConfig.from_yaml(Path(__file__).parent / "modal_cfg.yaml")
cuda_image = (
    modal.Image.from_dockerfile("Dockerfile")
    .entrypoint([])
    .env(
        {
            "TORCH_EXTENSIONS_DIR": cfg.cuda_kernel_cache_path,
            "TRITON_CACHE_DIR": cfg.triton_kernel_cache_path,
        }
    )
    .uv_sync()
    .add_local_dir(
        "./src/kompass",
        copy=False,
        remote_path=cfg.kernel_remote_path,
        ignore=["*.venv", "**/.DS_Store", "**/__pycache__"],
    )
)
app = modal.App("cuda-compile-and-run", image=cuda_image)
cuda_vol = modal.Volume.from_name(cfg.cuda_volume_name)
triton_vol = modal.Volume.from_name(cfg.triton_volume_name)


@app.function(
    image=cuda_image,
    volumes={
        cfg.cuda_kernel_cache_path: cuda_vol,
        cfg.triton_kernel_cache_path: triton_vol,
    },
    timeout=cfg.timeout,
)
def run_modal_bench(
    payload: Payload,
    progress: modal.Queue,
    rep_ms: int = 100,
) -> BenchResult:
    """Runs one sweep, pushing (`run_id`, container) to `progress` after each
    case."""
    from functools import partial

    from ..runner import run_bench

    message = (payload.run_id, os.environ.get("MODAL_TASK_ID", "modal"))
    return run_bench(
        payload, rep_ms=rep_ms, on_case_done=partial(progress.put, message)
    )


@app.function(
    image=cuda_image,
    volumes={
        cfg.cuda_kernel_cache_path: cuda_vol,
        cfg.triton_kernel_cache_path: triton_vol,
    },
    timeout=cfg.timeout,
)
def run_modal_tune(
    payload: Payload,
    progress: modal.Queue,
    rep_ms: int = 100,
) -> BenchResult:
    """Tunes one group of cases, pushing (`run_id`, container) to `progress`
    after each stage-1 config."""
    from functools import partial

    from ..runner import run_tune

    message = (payload.run_id, os.environ.get("MODAL_TASK_ID", "modal"))
    return run_tune(payload, rep_ms=rep_ms, on_case_done=partial(progress.put, message))


def run_modal(
    payloads: list[Payload],
    on_case_done: Callable[[str, str], None],
    gpu: str,
    kind: Literal["bench", "tune"],
    rep_ms: int = 100,
) -> list[BenchResult | BaseException]:
    """Parallel, serverless runner. Containers push (`run_id`, container id) to
    a queue after each case, a background thread drains it into `on_case_done`."""

    def drain(queue: modal.Queue, done: threading.Event) -> None:
        while not done.is_set():
            try:
                messages = queue.get_many(100, timeout=1)
            except Empty:
                continue
            for run_id, worker in messages:
                on_case_done(run_id, worker)
        while messages := queue.get_many(100, block=False):
            for run_id, worker in messages:
                on_case_done(run_id, worker)

    with (
        # modal.enable_output(),
        app.run(),
        modal.Queue.ephemeral() as queue,
    ):
        done = threading.Event()
        drainer = threading.Thread(target=drain, args=(queue, done))
        drainer.start()

        if kind == "bench":
            fn = run_modal_bench
        elif kind == "tune":
            fn = run_modal_tune
        else:
            raise ValueError(
                f"Unkown `kind` parameter, expected one of ['bench', 'tune'], got {kind}"
            )
        try:
            return list(
                fn.with_options(gpu=gpu, max_containers=cfg.max_containers).map(
                    payloads,
                    kwargs={"progress": queue, "rep_ms": rep_ms},
                    return_exceptions=True,
                )
            )
        finally:
            done.set()
            drainer.join()
