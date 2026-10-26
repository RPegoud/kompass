"""CPU-only fixtures: no test needs a GPU. The GPU-facing calls of the runner
(`describe_env`, `bench_copy`, `run_bench_case`) are replaced by fakes with
predictable timings, and every output (logs, tuned configs, reports) goes to a
temporary directory."""

from dataclasses import dataclass, field

import pytest

from kompass import runner
from kompass.harness import tuning, utils
from kompass.harness.schema import Case, CaseResult
from kompass.harness.utils import resolve_dtype
from kompass.kernels import REGISTRY

FAKE_GPU = "NVIDIA H100 80GB HBM3"  # a GPU `report.VENDOR_PEAKS` knows
SHARED_MEMORY = 232448  # H100's shared_memory_per_block_optin


def fake_ms(case: Case) -> float:
    """Default configs run at 10 TFLOPS. A tuned config runs faster the smaller
    its values, so the best config of a grid is its smallest values."""
    ms = REGISTRY[case.op].op.flops(case.params) / 10e12 * 1e3
    if case.tuning:
        ms *= 0.1 + sum(case.tuning.values()) / 1000
    return ms


def fake_result(case: Case, ms: float | None, status: str = "ok") -> CaseResult:
    spec = REGISTRY[case.op].impls[case.impl]
    op = REGISTRY[case.op].op
    dtype = resolve_dtype(case.dtype)
    return CaseResult(
        op=case.op,
        impl=case.impl,
        framework=spec.loader.framework,
        math=spec.math(dtype, case.options),
        params=case.params,
        options=case.options,
        tuning=case.tuning,
        dtype=str(dtype),
        seed=case.seed,
        quantiles=case.quantiles,
        flops=op.flops(case.params),
        bytes_moved=op.bytes_moved(case.params, dtype),
        device="cuda:0",
        times=None if ms is None else (ms * 0.9, ms, ms * 1.1),
        max_abs_diff=None if ms is None else 0.0,
        status=status,
        free_mem=0,
        reserved_mem=0,
        total_mem=0,
        required_mem=0,
    )


@dataclass
class FakeGPU:
    """What the fake `run_bench_case` saw: (case, rep_ms, abort_above_ms)."""

    calls: list[tuple[Case, int, float | None]] = field(default_factory=list)


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch):
    """Logs, reports and tuned configs go under `tmp_path`, not the repo."""
    monkeypatch.setattr(utils, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tuning, "TUNING_DIR", tmp_path / "tuning")
    return tmp_path


@pytest.fixture
def fake_gpu(monkeypatch) -> FakeGPU:
    gpu = FakeGPU()
    monkeypatch.setattr(
        runner,
        "describe_env",
        lambda: {
            "device_name": FAKE_GPU,
            "shared_memory_per_block_optin": SHARED_MEMORY,
        },
    )
    monkeypatch.setattr(runner, "bench_copy", lambda device, dtype: 3000.0)

    def run_bench_case(
        case, device, rep_ms=100, on_case_done=None, abort_above_ms=None
    ):
        gpu.calls.append((case, rep_ms, abort_above_ms))
        if on_case_done is not None:
            on_case_done()
        ms = fake_ms(case)
        if abort_above_ms is not None and ms > abort_above_ms:  # first call too slow
            return fake_result(case, None, status="slow")
        return fake_result(case, ms)

    monkeypatch.setattr(runner, "run_bench_case", run_bench_case)
    return gpu
