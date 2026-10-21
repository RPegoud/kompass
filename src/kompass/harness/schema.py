from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SweepConfig:
    run_id: str
    op: str
    seed: int
    impls: list[str]
    quantiles: tuple[float, ...]
    dtypes: list[str]
    shapes: list[dict[str, int]] | None = None
    axes: dict[str, list[int]] | None = None

    def __post_init__(self):
        if bool(self.axes) == bool(self.shapes):
            raise ValueError(
                f"Exactly one of `axes` or `shapes` should be defined. Got: {self.axes=} and {self.shapes=}"
            )


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
    op: str  # registry key, e.g. `vec_add`
    impl: str  # registry key, e.g. `vec_add_triton`
    framework: Literal["torch", "triton", "cuda"]
    math: str  # see KernelSpec.math
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
