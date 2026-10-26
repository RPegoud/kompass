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


def matmul(
    X: torch.Tensor, Y: torch.Tensor, input_precision: str = "ieee"
) -> torch.Tensor:
    """`torch.matmul` with float32 inputs multiplied as `input_precision`.
    cuBLAS reads it from a global flag, so it is set and restored around the call."""
    flags = torch.backends.cuda.matmul
    prev = flags.fp32_precision
    flags.fp32_precision = input_precision
    try:
        return torch.matmul(X, Y)
    finally:
        flags.fp32_precision = prev


def prune_matmul_config(
    params: dict[str, int],
    tuning: dict[str, int],
    dtype: torch.dtype,
    limits: dict[str, int],
) -> tuple[bool, str | None]:
    from triton import next_power_of_2

    t = tuning
    smem = (
        t["num_stages"]
        * (t["BLOCK_SIZE_M"] + t["BLOCK_SIZE_N"])
        * t["BLOCK_SIZE_K"]
        * dtype.itemsize
    )
    if smem > 0.9 * limits["shared_memory_per_block_optin"]:
        return True, f"needs ~{smem // 1024}KB of shared memory"

    for param, p_value in tuning.items():
        p_value = int(p_value)
        if "BLOCK_SIZE" in param:
            _, suffix = param.split("BLOCK_SIZE_", 2)
            if p_value < 16 or p_value > next_power_of_2(params[suffix]):
                return (
                    True,
                    f"`tl.dot` requires BLOCK_SIZE_M >= 16 and BLOCK_SIZE shouldn't exceed `next_power_of_two(X)`, got {p_value}",
                )

    return False, None


MatmulSpec = partial(KernelSpec, op=Matmul())
tiled_matmul_search_space = {
    "BLOCK_SIZE_M": [64, 128, 256],
    "BLOCK_SIZE_N": [64, 128, 256],
    "BLOCK_SIZE_K": [32, 64, 128],
    "num_warps": [4, 8],
    "num_stages": [2, 3, 4],
}

matmul_torch = MatmulSpec(
    loader=TorchLoader(fn=matmul), option_names=("input_precision",)
)

matmul_triton_tiled = MatmulSpec(
    loader=TritonLoader(
        module_path=f"{__package__}.matmul",
        fn_name="matmul_triton_tiled",
        supports_backward=False,
    ),
    option_names=("input_precision",),
    search_space=tiled_matmul_search_space,
    prune=prune_matmul_config,
)

matmul_triton_tiled_l2_swizzle = MatmulSpec(
    loader=TritonLoader(
        module_path=f"{__package__}.matmul",
        fn_name="matmul_triton_tiled_l2_swizzle",
        supports_backward=False,
    ),
    option_names=("input_precision",),
    search_space=tiled_matmul_search_space | {"GROUP_SIZE_M": [2, 4, 6]},
    prune=prune_matmul_config,
)

REGISTRY: dict[str, OpEntry] = {
    "matmul": OpEntry(
        op=Matmul(),
        impls={
            "matmul_triton_tiled": matmul_triton_tiled,
            "matmul_triton_tiled_l2_swizzle": matmul_triton_tiled_l2_swizzle,
            "matmul_torch": matmul_torch,
        },
    )
}
