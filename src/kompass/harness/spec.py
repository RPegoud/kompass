from collections.abc import Callable
from dataclasses import dataclass

import torch
from triton.testing import do_bench

from .loaders import Loader
from .ops import Op, TestResult


@dataclass(frozen=True)
class TestResults:
    correct: bool
    max_abs_diff: float
    dtype: str
    params: dict[str, int]
    seed: int


@dataclass(frozen=True)
class BenchResults:
    kernel_name: str
    params: dict[str, int]
    dtype: str
    seed: int
    times: dict[str, list[float]] | list[float]
    flops: int
    bytes_moved: int
    device: str


class KernelSpec:
    def __init__(self, op: Op, loader: Loader) -> None:
        self.op = op
        self.loader = loader
        self.name = self.loader.fn_name
        self._kernel = None

    @property
    def kernel(self) -> Callable:
        if self._kernel is None:
            self._kernel = self.loader.load()
        return self._kernel

    def _check(
        self,
        tensors: tuple[torch.Tensor, ...],
        *,
        params: dict[str, int],
        dtype: torch.dtype,
        seed: int,
    ) -> TestResults:
        pred = self.kernel(*tensors)
        ref = self.op.reference(*tensors)
        res: TestResult = self.op.compare(pred, ref)
        return TestResults(
            correct=res.correct,
            max_abs_diff=res.max_abs_diff,
            dtype=str(dtype),
            params=params,
            seed=seed,
        )

    def _make_inputs(
        self,
        params: dict[str, int],
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
    ) -> tuple[torch.Tensor, ...]:
        return self.op.make_inputs(
            params=params,
            dtype=dtype,
            requires_grad=self.loader.supports_backward,
            seed=seed,
            device=device,
        )

    def test(
        self,
        params: dict[str, int],
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
    ) -> TestResults:
        tensors = self.op.make_inputs(
            params=params,
            dtype=dtype,
            requires_grad=self.loader.supports_backward,
            seed=seed,
            device=device,
        )
        return self._check(tensors, params=params, dtype=dtype, seed=seed)

    def bench(
        self,
        params: dict[str, int],
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
        warmup: int = 25,
        rep: int = 100,
        quantiles: tuple[float, ...] = (0.2, 0.5, 0.8),
    ) -> BenchResults:
        tensors = self._make_inputs(
            params=params, dtype=dtype, device=device, seed=seed
        )

        gate = self._check(tensors=tensors, params=params, dtype=dtype, seed=seed)
        if not gate.correct:
            raise ValueError(
                f"The kernel did not pass the correctness test: {gate.correct=}, {gate.max_abs_diff=}\nAborting benchmark..."
            )

        times: list[float] = do_bench(
            fn=lambda: self.kernel(*tensors),
            warmup=warmup,
            rep=rep,
            quantiles=quantiles,
        )  # type: ignore

        return BenchResults(
            kernel_name=self.name,
            params=params,
            dtype=str(dtype),
            seed=seed,
            times=times,
            flops=self.op.flops(params, dtype),
            bytes_moved=self.op.bytes_moved(params, dtype),
            device=str(device),
        )
