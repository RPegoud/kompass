from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .loaders import Loader
from .ops import CompareResult, Op

if TYPE_CHECKING:
    import torch


class TooSlow(Exception):
    """A tuning config whose first call is already far slower than the best
    config found so far, so timing it in full isn't worth it."""


@dataclass(frozen=True)
class TestResults:
    correct: bool
    max_abs_diff: float
    dtype: str
    params: dict[str, int]
    seed: int


class KernelSpec:
    def __init__(
        self,
        op: Op,
        loader: Loader,
        option_names: tuple[str, ...] = (),
        search_space: dict[str, list[int]] | None = None,
        prune: Callable | None = None,
    ) -> None:
        self.op = op
        self.loader = loader
        self.option_names = option_names  # kwargs the kernel accepts, see Op
        self.name = self.loader.fn_name
        self._kernel = None
        self.search_space = search_space
        self.prune = prune

    def math(self, dtype: torch.dtype, options: dict[str, str] | None = None) -> str:
        """The arithmetic this kernel runs on `dtype`, which sets its compute
        ceiling: tensor-core ops run narrow dtypes natively and float32 as
        tf32 when `input_precision="tf32"`; other ops compute in float32 on
        CUDA cores."""
        if not self.op.tensor_cores:
            return "float32"
        if str(dtype) == "torch.float32":
            tf32 = (options or {}).get("input_precision") == "tf32"
            return "tf32" if tf32 else "float32"
        return str(dtype).removeprefix("torch.")

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
        kwargs: dict[str, Any] | None = None,
    ) -> TestResults:
        pred = self.kernel(*tensors, **(kwargs or {}))
        ref = self.op.reference(*tensors, params=params)
        res: CompareResult = self.op.compare(pred, ref)
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
        quantiles: tuple[float, ...] = (0.2, 0.5, 0.8),
        options: dict[str, str] | None = None,
        tuning: dict[str, int] | None = None,
        rep_ms: int = 100,
        abort_above_ms: float | None = None,
    ) -> tuple[tuple[float, ...], float]:
        """Check correctness, then time with `do_bench`. With `abort_above_ms`,
        one call is timed first and `TooSlow` is raised if it exceeds it."""
        tensors = self._make_inputs(
            params=params, dtype=dtype, device=device, seed=seed
        )
        kwargs = (options or {}) | (tuning or {})  # both reach the kernel as kwargs

        gate = self._check(
            tensors=tensors, params=params, dtype=dtype, seed=seed, kwargs=kwargs
        )
        if not gate.correct:
            raise ValueError(
                f"The kernel {self.name} did not pass the correctness test with params {params=}; {dtype=}: {gate.correct=}, {gate.max_abs_diff=}\nAborting benchmark..."
            )

        if abort_above_ms is not None:  # the gate already compiled the kernel
            import torch

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.kernel(*tensors, **kwargs)
            end.record()
            torch.cuda.synchronize()
            if (ms := start.elapsed_time(end)) > abort_above_ms:
                raise TooSlow(f"one call took {ms:.3g} ms > {abort_above_ms:.3g} ms")

        from triton.testing import do_bench

        times: tuple[float, ...] = do_bench(
            fn=lambda: self.kernel(*tensors, **kwargs),
            quantiles=quantiles,
            rep=rep_ms,
        )  # type: ignore

        return times, gate.max_abs_diff
