from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass()
class Run:
    """One sweep of a `bench` invocation: the `Payload` to execute, plus the
    bookkeeping (config name, run id, timestamp, log dir) the CLI needs to
    write its results to `logs/`. Built per config name before dispatch."""

    config_name: str
    run_id: str
    timestamp: str
    output_dir: Path
    payload: Payload


@dataclass()
class BenchResult:
    """What a runner returns for one `Run`: per-case result rows (`CaseResult`
    as dicts), run-level metadata, and the device/software environment. The
    CLI flattens it into `meta.json` and `times.jsonl`."""

    case_results: list[dict[str, Any]]
    meta: dict[str, Any]
    env: dict[str, str | int | None]


@dataclass(frozen=True)
class SweepConfig:
    """A sweep definition loaded from `cases/<config_name>.yaml`: an op, the
    impls and dtypes to compare, either an `axes` grid or an explicit
    `shapes` list, and optional kernel `options` to sweep. Expanded by `bench`
    into one `Case` per combination."""

    run_id: str
    op: str
    seed: int
    impls: list[str]
    quantiles: tuple[float, ...]
    dtypes: list[str]
    shapes: list[dict[str, int]] | None = None
    axes: dict[str, list[int]] | None = None
    options: dict[str, list[str]] | None = None  # e.g. {input_precision: [ieee, tf32]}

    def __post_init__(self):
        if bool(self.axes) == bool(self.shapes):
            raise ValueError(
                f"Exactly one of `axes` or `shapes` should be defined. Got: {self.axes=} and {self.shapes=}"
            )


@dataclass()
class Case:
    """A single benchmark unit: one {impl, params, dtype, options} combination
    of an op, with the seed and timing quantiles to use. Built from a
    `SweepConfig`."""

    op: str
    impl: str
    params: dict[str, int]
    dtype: str
    seed: int
    quantiles: tuple[float, ...]
    tuning: dict[str, int] = field(default_factory=dict)  # tuning parameters
    options: dict[str, str] = field(default_factory=dict)  # kernel kwargs


@dataclass()
class Payload:
    """Everything a runner needs to execute a sweep: its `Case`s and the target
    device. Serializable, so it is sent as-is to the Modal backend."""

    run_id: str
    op: str
    cases: list[Case]
    device: str


@dataclass(frozen=True)
class CaseResult:
    """The outcome of running one `Case`: status, timings, correctness diff,
    op cost model (flops, bytes) and memory readings. Becomes one row of
    `times.jsonl`."""

    op: str  # registry key, e.g. `vec_add`
    impl: str  # registry key, e.g. `vec_add_triton`
    framework: Literal["torch", "triton", "cuda"]
    math: str  # see KernelSpec.math
    params: dict[str, int]
    options: dict[str, str]
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
    tuning_parameters: dict[str, int] = None
    error: str | None = None


@dataclass
class ProfileRecord:
    """One kernel launch measured by Nsight Compute in a `profile` run, with
    the profiled case's metadata. Becomes one row of `records.jsonl`."""

    meta: dict[str, str | int]
    kernel_name: str
    action_idx: int
    duration_ns: float
    dram_bytes: int
    dram_pct_peak: float
    sm_pct_peak: float
