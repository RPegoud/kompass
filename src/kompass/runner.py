from dataclasses import asdict, dataclass
from typing import Any

import torch
import triton
import triton.language as tl
from tqdm.auto import tqdm
from triton.testing import do_bench

from .harness.case import Case
from .harness.spec import BenchResults, KernelSpec
from .harness.utils import describe_env, resolve_dtype
from .kernels import REGISTRY


@dataclass()
class Payload:
    run_id: str
    op: str
    cases: list[Case]
    device: str


@triton.jit
def copy_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def bench_copy(
    device: torch.device,
    n_bytes: int = 256 * 1024 * 1024 * 4,
    dtype: torch.dtype = torch.float32,
):
    a = torch.empty(n_bytes // dtype.itemsize, dtype=dtype, device=device)
    b = torch.empty_like(a)
    grid = lambda meta: (triton.cdiv(a.numel(), meta["BLOCK_SIZE"]),)
    ms: float = do_bench(
        lambda: copy_kernel[grid](
            in_ptr=a,
            out_ptr=b,
            n_elements=a.numel(),
            BLOCK_SIZE=256,  # type: ignore
        ),
        return_mode="max",
    )
    gb = 2 * n_bytes / 1e9  # copy_ moves bytes twice: read + write, multiply by 2
    return gb / (ms * 1e-3)


def run_bench(
    payload: Payload,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str | int | None]]:
    cases = payload.cases
    dtypes = [resolve_dtype(c.dtype) for c in cases]
    device = torch.device(payload.device)

    cp_bandwidth_triton = {str(t): bench_copy(device, dtype=t) for t in dtypes}

    env = describe_env()
    meta: dict[str, Any] = {
        "run_id": payload.run_id,
        "measured_copy_bandwidth_gbps_triton": cp_bandwidth_triton,
        "payload": asdict(payload),
    } | env
    times: list[dict[str, Any]] = []

    pbar = tqdm(cases)

    for case in pbar:
        pbar.set_description(f"Benchmarking {case.impl} with {case.params=}")
        spec: KernelSpec = REGISTRY[case.op].impls[case.impl]
        bench_res: BenchResults = spec.bench(
            params=case.params,
            dtype=resolve_dtype(case.dtype),
            device=device,
            seed=case.seed,
        )

        times.append(asdict(bench_res))

    return meta, times, env
