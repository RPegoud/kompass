from functools import partial

import torch
import torch.nn.functional as F

from ...harness.loaders import TorchLoader, TritonLoader
from ...harness.ops import Op, TestResult
from ...harness.spec import KernelSpec
from ...kernels import OpEntry


class Softmax(Op):
    param_names: tuple[str, ...] = ("M", "N")

    @staticmethod
    def reference(
        *tensors: torch.Tensor,
        params: dict[str, int],
    ) -> torch.Tensor:
        x = tensors[0]
        dtype = x.dtype
        ref = F.softmax(x, dim=-1, dtype=torch.float32)  # compute ref in full precision
        return ref.to(dtype)

    def compare(
        self,
        pred: torch.Tensor,
        ref: torch.Tensor,
    ) -> TestResult:
        correct = True
        try:
            torch.testing.assert_close(pred, ref)
        except AssertionError:
            correct = False

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
        gen = torch.Generator(device).manual_seed(seed)
        X = torch.randn(
            size=(params["M"], params["N"]),
            generator=gen,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return (X,)

    @staticmethod
    def bytes_moved(params: dict[str, int], dtype: torch.dtype) -> int:
        """Softmax must move at least 2MN bytes."""
        return params["M"] * params["N"] * dtype.itemsize * 2

    @staticmethod
    def flops(params: dict[str, int]) -> int:
        """
        The theoretical amount of FLOPs is 3N for softmax, although in practice
        computing the max for numerical stability raises FLOPs to 5N.
        We report the algorithmic ideal.
        """
        return params["M"] * params["N"] * 3


SoftmaxSpec = partial(KernelSpec, op=Softmax())

softmax_triton_single_block = SoftmaxSpec(
    loader=TritonLoader(
        module_path=f"{__package__}.softmax",
        fn_name="softmax_triton_single_block",
        supports_backward=False,
    ),
)
softmax_triton_online = SoftmaxSpec(
    loader=TritonLoader(
        module_path=f"{__package__}.softmax",
        fn_name="softmax_triton_online",
        supports_backward=False,
    ),
)
softmax_torch = SoftmaxSpec(loader=TorchLoader(torch.softmax, options={"dim": -1}))

REGISTRY: dict[str, OpEntry] = {
    "softmax": OpEntry(
        op=Softmax(),
        impls={
            "softmax_triton_single_block": softmax_triton_single_block,
            "softmax_triton_online": softmax_triton_online,
            "softmax_torch": softmax_torch,
        },
    )
}
