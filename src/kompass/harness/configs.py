from dataclasses import dataclass


@dataclass(frozen=True)
class SweepConfig:
    run_id: str
    op: str
    seed: int
    impls: list[str]
    axes: dict[str, list]
    quantiles: list[float]
