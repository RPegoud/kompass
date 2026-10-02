import torch

from .spec import KernelSpec


def profile_one(
    spec: KernelSpec,
    params: dict[str, int],
    dtype: torch.dtype,
    seed: int,
    device: torch.device,
    warmup: int = 5,
):
    tensors = spec.op.make_inputs(
        params=params,
        dtype=dtype,
        requires_grad=spec.loader.supports_backward,
        device=device,
        seed=seed,
    )

    for _ in range(warmup):
        spec.kernel(*tensors)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    spec.kernel(*tensors)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
