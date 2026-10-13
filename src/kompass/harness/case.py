from dataclasses import dataclass


@dataclass()
class Case:
    op: str
    impl: str
    params: dict[str, int]
    dtype: str
    seed: int
    quantiles: tuple[float, ...]
