"""Plain data flowing through a run: sweep config -> cases -> payload -> results.

Keep this module free of torch/triton imports: `Payload` is pickled to Modal.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SweepConfig:
    run_id: str
    op: str
    seed: int
    impls: list[str]
    axes: dict[str, list]
    quantiles: tuple[float, ...]


@dataclass()
class Case:
    op: str
    impl: str
    params: dict[str, int]
    dtype: str
    seed: int
    quantiles: tuple[float, ...]


@dataclass()
class Payload:
    run_id: str
    op: str
    cases: list[Case]
    device: str


@dataclass(frozen=True)
class CaseResult:
    impl: str
    params: dict[str, int]
    dtype: str
    seed: int
    quantiles: tuple[float, ...]
    flops: int
    bytes_moved: int
    device: str
    times: tuple[float, ...] | None
    max_abs_diff: float | None
    status: str
    free_mem: int
    reserved_mem: int
    total_mem: int
    required_mem: int
    error: str | None = None


@dataclass
class ProfileRecord:
    meta: dict[str, str | int]
    kernel_name: str
    action_idx: int
    duration_ns: float
    dram_bytes: int
    dram_pct_peak: float
    sm_pct_peak: float
