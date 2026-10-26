from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from typing import Any, Literal

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

from .harness.ops import Op
from .harness.schema import BenchResult, Case, CaseResult, Payload
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


def run_bench_case(
    case: Case,
    device: torch.device,
    rep_ms: int = 100,
    coarse_rep_ratio: float = 1.0,
    on_case_done: Callable[[], None] | None = None,
) -> CaseResult:
    spec: KernelSpec = REGISTRY[case.op].impls[case.impl]
    op: Op = REGISTRY[case.op].op
    dtype = resolve_dtype(case.dtype)

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
                options=case.options,
                rep_ms=rep_ms,
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

    case_res = CaseResult(
        op=case.op,
        impl=case.impl,
        framework=spec.loader.framework,
        math=spec.math(dtype, case.options),
        params=case.params,
        options=case.options,
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

    torch.cuda.empty_cache()
    if on_case_done is not None:
        on_case_done()
    return case_res


def run_bench(
    payload: Payload, rep_ms: int = 100, on_case_done: Callable[[], None] | None = None
) -> BenchResult:
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
    case_results: list[dict[str, Any]] = []
    for case in cases:
        case_res = asdict(
            run_bench_case(
                case, device=device, rep_ms=rep_ms, on_case_done=on_case_done
            )
        )
        case_results.append(case_res)

    return BenchResult(
        case_results=case_results,
        meta=meta,
        env=env,
    )


def run_tune(
    payload: Payload,
    rep_ms: int,
    coarse_rep_ratio: float = 0.25,  # proportion of time to spend per case during the coarse run
    coarse_candidate_ratio: float = 0.1,  # proportion of candidates from the coarse sweep to keep for the final run
    on_case_done: Callable[[], None] | None = None,
) -> BenchResult:

    def _prune_cases(
        cases: list[Case], env: dict[str, str | int | None]
    ) -> list[Case | str]:
        pruned = []
        for case in cases:
            spec = REGISTRY[case.op].impls[case.impl]
            if spec.prune:
                prune_case, msg = spec.prune(
                    params=case.params,
                    tuning=case.tuning,
                    limits=env,
                    dtype=resolve_dtype(case.dtype),
                )
                if prune_case:
                    pruned.append(msg)
                    continue
            pruned.append(case)
        return pruned

    cases = payload.cases
    dtypes = {resolve_dtype(c.dtype) for c in cases}
    device = torch.device(payload.device)
    env = describe_env()

    cp_bandwidth_triton = {str(t): bench_copy(device, dtype=t) for t in dtypes}

    pruned: list[Case, str] = _prune_cases(cases=cases, env=env)

    meta: dict[str, Any] = {
        "run_id": payload.run_id,
        "measured_copy_bandwidth_gbps_triton": cp_bandwidth_triton,
        "payload": asdict(payload),
    } | env
    case_results: list[dict[str, Any]] = []

    for case in pruned:
        if isinstance(case, str):
            case_results.append({"skipped": case})
            continue
        case_res = asdict(
            run_bench_case(
                case,
                rep_ms=rep_ms,
                coarse_rep_ratio=coarse_rep_ratio,
                device=device,
                on_case_done=on_case_done,
            )
        )
        case_results.append(case_res)

    groups = defaultdict(dict[dict])  # [(impl, math, dtype): {params: (tuning, times)}]
    for res in case_results:
        groups[(res["impl"], res["math"], res["dtype"])][
            tuple(res["params"].items())
        ] = (
            res["tuning"],
            res["quantiles"],
        )

    candidate_tunings: dict[tuple, list] = {}
    for group in groups:  # grab the best N tunings per
        for params, values in group.items():
            sorted_tunings = sorted((params, values), key=lambda t: t[1])
            n_candidates = len(sorted_tunings)
            candidate_configs = sorted_tunings[
                : int(n_candidates * coarse_candidate_ratio)
            ]
            candidate_tunings[params].append(candidate_configs)

    return BenchResult(
        case_results=case_results,
        meta=meta,
        env=env,
    )


def run_local(
    payloads: list[Payload],
    on_case_done: Callable[[str], None],
    kind: Literal["bench", "tune"],
    rep_ms: int = 100,
    coarse_rep_ratio: float = 1.0,
) -> list[BenchResult | BaseException]:
    """Serial, local runner."""
    results: list[BenchResult | BaseException] = []
    if kind == "bench":
        fn = run_bench
    elif kind == "tune":
        fn = partial(run_tune, coarse_rep_ratio=coarse_rep_ratio)
    else:
        raise ValueError(
            f"Unkown `kind` parameter, expected one of ['bench', 'tune'], got {kind}"
        )

    for payload in payloads:
        try:
            results.append(
                fn(
                    payload,
                    rep_ms=rep_ms,
                    on_case_done=partial(on_case_done, payload.run_id),
                )
            )
        except Exception as e:  # noqa: BLE001
            results.append(e)
    return results


def select_runner(
    backend: str, kind: Literal["bench", "tune"]
) -> Callable[list[Payload], Callable[[str], None], Literal["bench", "true"], int]:
    import typer

    if backend == "local":
        runner = partial(run_local, kind=kind)

    elif backend.startswith("modal:"):
        *_, gpu = backend.partition(":")
        gpu = gpu.upper()
        if gpu == "H100":
            gpu += "!"  # `!` prevents automatic upgrades to H200

        from .infra.modal_runner import run_modal

        runner = partial(run_modal, gpu=gpu, kind=kind)

    else:
        raise typer.BadParameter(
            f"Expected backend format `modal:gpu`, e.g. `modal:h100`, got {backend}"
        )

    return runner
