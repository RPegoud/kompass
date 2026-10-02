from pathlib import Path

import torch

from ...harness.loaders import CudaLoader, TorchLoader, TritonLoader
from ...harness.ops import Op, TestResult
from ...harness.spec import KernelSpec
from ...kernels import OpEntry


class VecAdd(Op):
    param_names: tuple[str, ...] = ("N",)

    @staticmethod
    def reference(*tensors: torch.Tensor) -> torch.Tensor:
        A, B = tensors
        return torch.add(A, B)

    @staticmethod
    def compare(pred: torch.Tensor, ref: torch.Tensor) -> TestResult:
        return TestResult(
            correct=torch.equal(pred, ref),
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
        A = torch.randn(
            *n,
            generator=gen,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        B = torch.randn(
            *n,
            generator=gen,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return A, B

    @staticmethod
    def bytes_moved(params: dict[str, int], dtype: torch.dtype) -> int:
        return 3 * params["N"] * dtype.itemsize

    @staticmethod
    def flops(params: dict[str, int], dtype: torch.dtype) -> int:
        return params["N"]


vec_add_torch = KernelSpec(
    op=VecAdd(),
    loader=TorchLoader(fn=torch.add),
)
vec_add_cuda_v1 = KernelSpec(
    op=VecAdd(),
    loader=CudaLoader(
        module_name="vec_add_cuda_v1",
        fn_name="vec_add_cuda_v1",
        sources=[str(Path(__file__).parent / "vecAdd.cu")],
        supports_backward=False,
    ),
)
vec_add_triton_v1 = KernelSpec(
    op=VecAdd(),
    loader=TritonLoader(
        module_path=f"{__package__}.vec_add",
        fn_name="vec_add_triton_v1",
        supports_backward=False,
    ),
)

REGISTRY: dict[str, OpEntry] = {
    "vec_add": OpEntry(
        op=VecAdd(),
        impls={
            "vec_add_torch": vec_add_torch,
            "vec_add_cuda_v1": vec_add_cuda_v1,
            "vec_add_triton_v1": vec_add_triton_v1,
        },
    )
}
