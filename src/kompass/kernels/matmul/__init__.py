from functools import partial

import torch

from ...harness.loaders import TorchLoader, TritonLoader
from ...harness.ops import CompareResult, Op
from ...harness.spec import KernelSpec
from ...kernels import OpEntry


class Matmul(Op):
    param_names: tuple[str, ...] = ("M", "N", "K")
    option_names: tuple[str, ...] | None = ("input_precision",)
    tensor_cores = True

    @staticmethod
    def reference(
        *tensors: torch.Tensor,
        params: dict[str, int],
    ) -> torch.Tensor:
        X, Y = tensors
        dtype = X.dtype
        ref = torch.matmul(
            X.to(torch.float32), Y.to(torch.float32)
        )  # compute the ref in full precision
        return ref.to(dtype)

    def compare(
        self,
        pred: torch.Tensor,
        ref: torch.Tensor,
    ) -> CompareResult:
        """Use the Frobenius norm to compare the reference and predicted values."""
        norm = lambda x: torch.linalg.vector_norm(x, ord=2)
        correct = (norm(pred - ref) / norm(ref)) < 1e-2

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


MatmulSpec = partial(KernelSpec(op=Matmul()))

matmul_torch = MatmulSpec(loader=TorchLoader(fn=torch.matmul))

matmul_triton_tiled_ieee = MatmulSpec(
    loader=TritonLoader(
        module_path=f"{__package__}.matmul",
        fn_name="matmul_triton_tiled_ieee",
        supports_backward=False,
    ),
)
matmul_triton_tiled_tf32 = MatmulSpec(
    loader=TritonLoader(
        module_path=f"{__package__}.matmul",
        fn_name="matmul_triton_tiled_tf32",
        supports_backward=False,
    ),
    precision="tf32",
)
matmul_triton_tiled_l2_swizzle_ieee = MatmulSpec(
    loader=TritonLoader(
        module_path=f"{__package__}.matmul",
        fn_name="matmul_triton_tiled_l2_swizzle_ieee",
        supports_backward=False,
    ),
)
matmul_triton_tiled_l2_swizzle_tf32 = MatmulSpec(
    loader=TritonLoader(
        module_path=f"{__package__}.matmul",
        fn_name="matmul_triton_tiled_l2_swizzle_tf32",
        supports_backward=False,
    ),
    precision="tf32",
)

REGISTRY: dict[str, OpEntry] = {
    "matmul": OpEntry(
        op=Matmul(),
        impls={
            "matmul_triton_tiled_ieee": matmul_triton_tiled_ieee,
            "matmul_triton_tiled_tf32": matmul_triton_tiled_tf32,
            "matmul_triton_tiled_l2_swizzle_ieee": matmul_triton_tiled_l2_swizzle_ieee,
            "matmul_triton_tiled_l2_swizzle_tf32": matmul_triton_tiled_l2_swizzle_tf32,
            "matmul_torch": matmul_torch,
        },
    )
}
