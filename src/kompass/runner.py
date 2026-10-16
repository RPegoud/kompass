from dataclasses import asdict
from typing import Any

import torch
import triton
import triton.language as tl
from tqdm.auto import tqdm
from triton.testing import do_bench

from .harness.ops import Op
from .harness.schema import CaseResult, Payload
from .harness.spec import KernelSpec
from .harness.utils import describe_env, resolve_dtype
from .kernels import REGISTRY


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


def get_mem_reqs(
    op: Op, params: dict[str, int], dtype: torch.dtype, device: torch.device
) -> tuple[int, int, int, bool]:
    required = op.bytes_moved(params, dtype)
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    return free_mem, total_mem, required, required < 0.8 * free_mem


def run_bench(
    payload: Payload,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str | int | None]]:
    cases = payload.cases
    dtypes = {resolve_dtype(c.dtype) for c in cases}
    device = torch.device(payload.device)

    cp_bandwidth_triton = {str(t): bench_copy(device, dtype=t) for t in dtypes}

    env = describe_env()
    meta: dict[str, Any] = {
        "run_id": payload.run_id,
        "measured_copy_bandwidth_gbps_triton": cp_bandwidth_triton,
        "payload": asdict(payload),
    } | env
    sweep_times: list[dict[str, Any]] = []

    pbar = tqdm(cases)

    for case in pbar:
        spec: KernelSpec = REGISTRY[case.op].impls[case.impl]
        op: Op = REGISTRY[case.op].op
        dtype = resolve_dtype(case.dtype)
        pbar.set_description(f"Benchmarking {case.impl} with {case.params=}, {dtype=}")

        free_mem, total_mem, required_mem, enough_mem = get_mem_reqs(
            op=op, params=case.params, dtype=dtype, device=device
        )
        reserved_mem = torch.cuda.memory_reserved(device)
        status, error = "ok", None
        case_times, max_abs_diff = None, None

        if enough_mem:
            try:
                case_times, max_abs_diff = spec.bench(
                    params=case.params,
                    dtype=dtype,
                    device=device,
                    seed=case.seed,
                    quantiles=case.quantiles,
                )

            except ValueError as e:  # correctness error
                status = "incorrect"
                error = repr(e)

            except torch.cuda.OutOfMemoryError as e:
                status = "oom"
                error = repr(e)

            except Exception as e:  # noqa: BLE001
                status = "failed"
                error = repr(e)

        else:
            status = "skipped_oom"

        bench_res = CaseResult(
            impl=spec.name,
            params=case.params,
            dtype=str(dtype),
            seed=case.seed,
            times=case_times,
            max_abs_diff=max_abs_diff,
            quantiles=case.quantiles,
            flops=op.flops(case.params),
            bytes_moved=op.bytes_moved(case.params, dtype),
            device=str(device),
            status=status,
            error=error,
            free_mem=free_mem,
            reserved_mem=reserved_mem,
            total_mem=total_mem,
            required_mem=required_mem,
        )
        sweep_times.append(asdict(bench_res))
        torch.cuda.empty_cache()

    return meta, sweep_times, env
