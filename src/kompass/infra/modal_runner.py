from typing import Any

import modal

from ..runner import Payload

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
        ignore=["*.venv"],
    )
)
app = modal.App("cuda-compile-and-run", image=cuda_image)
cuda_vol = modal.Volume.from_name(CUDA_VOLUME_NAME)
triton_vol = modal.Volume.from_name(TRITON_VOLUME_NAME)


@app.function(
    image=cuda_image,
    volumes={CUDA_KERNEL_CACHE_PATH: cuda_vol, TRITON_KERNEL_CACHE_PATH: triton_vol},
)
def run_modal_bench(
    payload: Payload,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str | int | None]]:
    from ..runner import run_bench

    return run_bench(payload)
