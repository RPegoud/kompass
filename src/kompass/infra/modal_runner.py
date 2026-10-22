import threading
from collections.abc import Callable
from queue import Empty

import modal

from ..harness.schema import BenchResult, Payload

CUDA_VOLUME_NAME = "cuda_kernels"
TRITON_VOLUME_NAME = "triton_kernels"
KERNEL_REMOTE_PATH = "/kernels"
CUDA_KERNEL_CACHE_PATH = "/cache/cuda"
TRITON_KERNEL_CACHE_PATH = "/cache/triton"

cuda_image = (
    modal.Image.from_dockerfile("Dockerfile")
    .entrypoint([])
    .env(
        {
            "TORCH_EXTENSIONS_DIR": CUDA_KERNEL_CACHE_PATH,
            "TRITON_CACHE_DIR": TRITON_KERNEL_CACHE_PATH,
        }
    )
    .uv_sync()
    .add_local_dir(
        "./src/kompass",
        copy=False,
        remote_path=KERNEL_REMOTE_PATH,
        ignore=["*.venv", "**/.DS_Store", "**/__pycache__"],
    )
)
app = modal.App("cuda-compile-and-run", image=cuda_image)
cuda_vol = modal.Volume.from_name(CUDA_VOLUME_NAME)
triton_vol = modal.Volume.from_name(TRITON_VOLUME_NAME)


@app.function(
    image=cuda_image,
    volumes={CUDA_KERNEL_CACHE_PATH: cuda_vol, TRITON_KERNEL_CACHE_PATH: triton_vol},
    timeout=1800,  # 30 minutes timeout
)
def run_modal_bench(payload: Payload, progress: modal.Queue) -> BenchResult:
    """Runs one sweep, pushing its `run_id` to `progress` after each case."""
    from functools import partial

    from ..runner import run_bench

    return run_bench(payload, on_case_done=partial(progress.put, payload.run_id))


def run_modal(
    payloads: list[Payload], on_case_done: Callable[[str], None], gpu: str
) -> list[BenchResult | BaseException]:
    """Parallel, serverless runner. Containers push a `run_id` to a queue after
    each case, a background thread drains it into `on_case_done`."""

    def drain(queue: modal.Queue, done: threading.Event) -> None:
        while not done.is_set():
            try:
                run_ids = queue.get_many(100, timeout=1)
            except Empty:
                continue
            for run_id in run_ids:
                on_case_done(run_id)
        while run_ids := queue.get_many(100, block=False):
            for run_id in run_ids:
                on_case_done(run_id)

    with (
        # modal.enable_output(),
        app.run(),
        modal.Queue.ephemeral() as queue,
    ):
        done = threading.Event()
        drainer = threading.Thread(target=drain, args=(queue, done))
        drainer.start()
        try:
            return list(
                run_modal_bench.with_options(gpu=gpu).map(
                    payloads,
                    kwargs={"progress": queue},
                    return_exceptions=True,
                )
            )
        finally:
            done.set()
            drainer.join()
