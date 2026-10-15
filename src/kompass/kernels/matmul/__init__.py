from functools import partial

import torch

from ...harness.loaders import TorchLoader, TritonLoader
from ...harness.ops import CompareResult, Op
from ...harness.spec import KernelSpec
from ...kernels import OpEntry


class Matmul(Op):
    param_names: tuple[str, ...] = ("M", "N", "K")

    @staticmethod
    def reference(*tensors: torch.Tensor, params: dict[str, int]) -> torch.Tensor:
        X, Y = tensors
        dtype = X.dtype
        ref = torch.matmul(
            X.to(dtype), Y.to(dtype)
        )  # compute the ref in full precision
        return ref.to(dtype)

    def compare(self, pred: torch.Tensor, ref: torch.Tensor) -> CompareResult:
        correct = True
        try:
            torch.testing.assert_close(pred, ref)
        except AssertionError:
            correct = False

        return CompareResult(
            correct=correct, max_abs_diff=torch.max(torch.abs(ref - pred)).item()
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
            size=(params["M"], params["K"]),
            generator=gen,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        Y = torch.randn(
            size=(params["K"], params["N"]),
            generator=gen,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return X, Y

    @staticmethod
    def bytes_moved(params: dict[str, int], dtype: torch.dtype) -> int:
        M, N, K = params["M"], params["N"], params["K"]
        return (M * K + K * N + M * N) * dtype.itemsize

    @staticmethod
    def flops(params: dict[str, int]) -> int:
        M, N, K = params["M"], params["N"], params["K"]
        return 2 * M * N * K


MatmulSpec = partial(KernelSpec, op=Matmul())

matmul_torch = MatmulSpec(loader=TorchLoader(fn=torch.matmul))

matmul_triton_tiled = MatmulSpec(
    loader=TritonLoader(
        module_path=f"{__package__}.matmul",
        fn_name="triton_matmul_tiled",
        supports_backward=False,
    )
)

# matmul_cuda_naive = MatmulSpec(loader=CudaLoader(...))

REGISTRY: dict[str, OpEntry] = {
    "matmul": OpEntry(
        op=Matmul(),
        impls={
            "matmul_triton_tiled": matmul_triton_tiled,
            "matmul_torch": matmul_torch,
        },
    )
}
