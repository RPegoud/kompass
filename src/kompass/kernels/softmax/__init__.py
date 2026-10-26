from functools import partial

import torch
import torch.nn.functional as F

from ...harness.loaders import TorchLoader, TritonLoader
from ...harness.ops import CompareResult, Op
from ...harness.spec import KernelSpec
from ...kernels import OpEntry


class Softmax(Op):
    param_names: tuple[str, ...] = ("M", "N")
    option_names = None

    @staticmethod
    def reference(
        *tensors: torch.Tensor,
        params: dict[str, int],
    ) -> torch.Tensor:
        X = tensors[0]
        dtype = X.dtype
        ref = F.softmax(X, dim=-1, dtype=torch.float32)  # compute ref in full precision
        return ref.to(dtype)

    def compare(
        self,
        pred: torch.Tensor,
        ref: torch.Tensor,
        options: dict[str, int | str] | None = None,
    ) -> CompareResult:
        correct = True
        try:
            torch.testing.assert_close(pred, ref)
        except AssertionError:
            correct = False

        return CompareResult(
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


def prune_softmax_config(
    params: dict[str, int],
    tuning: dict[str, int],
    dtype: torch.dtype,
    limits: dict[str, int],
) -> tuple[bool, str | None]:
    """Shared memory is never the limit: a program holds one row (or one
    block of it) in registers. Only skip wasted blocks."""
    from triton import next_power_of_2

    block = tuning.get("BLOCK_SIZE")
    if block is None:  # single-block: the block is the whole row
        return False, None
    if block > next_power_of_2(params["N"]):
        return True, f"BLOCK_SIZE {block} > next_power_of_2(N={params['N']})"
    if block < 32 * tuning["num_warps"]:
        return (
            True,
            f"BLOCK_SIZE {block} leaves threads of {tuning['num_warps']} warps idle",
        )
    return False, None


SoftmaxSpec = partial(KernelSpec, op=Softmax())

softmax_triton_single_block = SoftmaxSpec(
    loader=TritonLoader(
        module_path=f"{__package__}.softmax",
        fn_name="softmax_triton_single_block",
        supports_backward=False,
    ),
    search_space={"num_warps": [4, 8, 16, 32]},
    prune=prune_softmax_config,
)
softmax_triton_online = SoftmaxSpec(
    loader=TritonLoader(
        module_path=f"{__package__}.softmax",
        fn_name="softmax_triton_online",
        supports_backward=False,
    ),
    search_space={
        "num_warps": [4, 8, 16],
        "num_stages": [2, 3, 4],
        "BLOCK_SIZE": [128, 256, 512, 1024, 2048, 4096, 8192],
    },
    prune=prune_softmax_config,
)
softmax_torch = SoftmaxSpec(
    loader=TorchLoader(torch.softmax, options={"dim": -1}),
)

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
