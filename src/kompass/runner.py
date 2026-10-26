import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from typing import Any, Literal

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

from .harness import tuning
from .harness.ops import Op
from .harness.schema import BenchResult, Case, CaseResult, Payload
from .harness.spec import KernelSpec, TooSlow
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
    on_case_done: Callable[[], None] | None = None,
    abort_above_ms: float | None = None,
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
                tuning=case.tuning,
                rep_ms=rep_ms,
                abort_above_ms=abort_above_ms,
            )

        except TooSlow as e:  # tuning: not worth timing in full
            status = "slow"
            error = repr(e)

        except ValueError as e:  # correctness error
            status = "incorrect"
            error = repr(e)

        except torch.cuda.OutOfMemoryError as e:
            status = "oom"
            error = repr(e)

        except triton.runtime.errors.OutOfResources as e:  # tuning: smem/registers
            status = "oom_smem"
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
        tuning=case.tuning,
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
    env = describe_env()
    cases = payload.cases
    if payload.tuning_mode != "untuned":  # on the worker: only it knows its GPU
        tuned = tuning.apply(cases, str(env["device_name"]))
        if payload.tuning_mode == "tuned":
            cases = tuned
        else:  # both: the defaults, plus a tuned twin of each case with a config
            cases = cases + [t for c, t in zip(cases, tuned) if t.tuning != c.tuning]
    dtypes = {resolve_dtype(c.dtype) for c in cases}
    device = torch.device(payload.device)

    cp_bandwidth_triton = {str(t): bench_copy(device, dtype=t) for t in dtypes}

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


def _group_key(case: Case) -> tuple:
    """Tuning configs only compete with configs of the same impl, dtype,
    options and shape."""
    return (
        case.impl,
        case.dtype,
        tuple(sorted(case.options.items())),
        tuple(sorted(case.params.items())),
    )


def _median_ms(case: Case, res: CaseResult) -> float:
    return tuning.median_ms(res.times, case.quantiles)  # type: ignore


def run_tune(
    payload: Payload,
    rep_ms: int = 100,
    coarse_rep_ms: int = 25,
    top_k: int = 5,
    slow_factor: float = 3.0,
    slow_floor_ms: float = 1.0,
    on_case_done: Callable[[], None] | None = None,
) -> BenchResult:
    """Time every tuning config briefly, then re-time the `top_k` fastest per
    group (`_group_key`) with the full `rep_ms`, on the same GPU. Rows carry
    `stage` 1 or 2; configs the kernel's `prune` rejects go to `meta["pruned"]`.
    `on_case_done` counts the configs of stage 1, pruned ones included.

    In stage 1, a config whose first call exceeds `slow_factor` x the group's
    best median so far is recorded as `slow` instead of being timed. The
    `slow_floor_ms` keeps small kernels, whose single calls are noisy, from
    being skipped: only configs that would cost real time are."""
    device = torch.device(payload.device)
    env = describe_env()

    # 0. drop configs the kernel says can't work, keeping the reason
    kept, pruned = [], []
    for case in payload.cases:
        spec = REGISTRY[case.op].impls[case.impl]
        if spec.prune:
            skip, reason = spec.prune(
                params=case.params,
                tuning=case.tuning,
                limits=env,
                dtype=resolve_dtype(case.dtype),
            )
            if skip:
                pruned.append(
                    {"params": case.params, "tuning": case.tuning, "reason": reason}
                )
                if on_case_done is not None:
                    on_case_done()
                continue
        kept.append(case)

    # 1. coarse pass over every remaining config
    coarse: list[tuple[Case, CaseResult]] = []
    best: dict[tuple, float] = {}
    for case in kept:
        key = _group_key(case)
        limit = max(slow_factor * best[key], slow_floor_ms) if key in best else None
        res = run_bench_case(
            case, device, coarse_rep_ms, on_case_done, abort_above_ms=limit
        )
        if res.status == "ok":
            best[key] = min(best.get(key, math.inf), _median_ms(case, res))
        coarse.append((case, res))

    # 2. the top_k fastest valid configs per group
    by_group = defaultdict(list)
    for case, res in coarse:
        if res.status == "ok":
            by_group[_group_key(case)].append((_median_ms(case, res), case))
    finalists = [
        case
        for group in by_group.values()
        for _, case in sorted(group, key=lambda t: t[0])[:top_k]
    ]

    # 3. fine pass on the finalists only
    fine = [run_bench_case(case, device, rep_ms) for case in finalists]

    rows = [asdict(res) | {"stage": 1} for _, res in coarse]
    rows += [asdict(res) | {"stage": 2} for res in fine]
    meta: dict[str, Any] = {
        "run_id": payload.run_id,
        "pruned": pruned,
        "payload": asdict(payload),
    } | env
    return BenchResult(case_results=rows, meta=meta, env=env)


def run_local(
    payloads: list[Payload],
    on_case_done: Callable[[str, str], None],
    kind: Literal["bench", "tune"],
    rep_ms: int = 100,
) -> list[BenchResult | BaseException]:
    """Serial, local runner."""
    results: list[BenchResult | BaseException] = []
    if kind == "bench":
        fn = run_bench
    elif kind == "tune":
        fn = run_tune
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
                    on_case_done=partial(on_case_done, payload.run_id, "local"),
                )
            )
        except Exception as e:  # noqa: BLE001
            results.append(e)
    return results


def select_runner(backend: str, kind: Literal["bench", "tune"]) -> Callable:
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
