import torch
import torch.nn.functional as F

from ...harness.ops import Op, TestResult


class Softmax(Op):
    param_names: tuple[str, ...] = ("N", "dim")

    def __init__(self) -> None:
        self.tolerance_per_dtype = {
            "bf16": {"atol": 1e-6, "rtol": 1e-3},
            "float32": {"atol": 1e-6, "rtol": 1e-3},
        }

    @staticmethod
    def reference(*tensors: torch.Tensor, params: dict[str, int]) -> torch.Tensor:
        x = tensors[0]
        return F.softmax(x, dim=params["dim"])

    def compare(self, pred: torch.Tensor, ref: torch.Tensor) -> TestResult:
        correct = False
        dtype = pred.dtype

        torch.testing.assert_close(
            pred,
            ref,
            rtol=self.tolerance_per_dtype[str(dtype)]["rtol"],
            atol=self.tolerance_per_dtype[str(dtype)]["atol"],
        )

        correct = True

        return TestResult(
            correct=correct,
            max_abs_diff=torch.max(torch.abs(ref - pred)).item(),
        )

    @staticmethod
    def make_inputs(
        params: dict[str, int],
        dtype: torch.dtype,
        requires_grad: bool,
        seed: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        n: tuple[int] = (params["N"],)
        gen = torch.Generator(device).manual_seed(seed)
        X = torch.randn(
            *n,
            generator=gen,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return (X,)

    @staticmethod
    def bytes_moved(params: dict[str, int], dtype: torch.dtype) -> int:
        return params["N"] * dtype.itemsize * 2

    @staticmethod
    def flops(params: dict[str, int], dtype: torch.dtype) -> int:
        """
        Theoretically, the softmax requires `3N` FLOPs, however numerically stable
        variatns require `5N` to find the max and subtract it from the input.
        """
        return params["N"] * 5
