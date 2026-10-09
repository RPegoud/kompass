import json
from pathlib import Path

import torch

from .spec import KernelSpec
from .utils import describe_env, resolve_dtype, resolve_impl


def profile_one(
    spec: KernelSpec,
    params: dict[str, int],
    dtype: torch.dtype,
    seed: int,
    device: torch.device,
    meta_file: Path,
    warmup: int = 5,
):
    try:
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

    except Exception as e:
        meta = json.loads(meta_file.read_text())
        meta |= {"error": repr(e)}
        meta_file.write_text(json.dumps(meta, indent=2))
        raise

    # the host might not have a GPU, record the env that actually ran the kernel
    env = describe_env()
    meta = json.loads(meta_file.read_text())
    meta |= env
    meta_file.write_text(json.dumps(meta, indent=2))


def main(meta_file: Path) -> None:
    meta = json.loads(meta_file.read_text())
    _, spec = resolve_impl(f"{meta['op']}/{meta['impl']}")
    profile_one(
        spec,
        params=meta["params"],
        dtype=resolve_dtype(meta["dtype"]),
        seed=meta["seed"],
        device=torch.device(meta["device"]),
        warmup=meta["warmup"],
        meta_file=meta_file,
    )


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]))