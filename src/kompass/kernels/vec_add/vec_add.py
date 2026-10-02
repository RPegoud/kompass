import torch
import triton
import triton.language as tl


def _check_vec_add_inputs(A: torch.Tensor, B: torch.Tensor) -> None:
    assert A.shape == B.shape, (
        f"Inputs must have the same shape, got {A.shape=} and {B.shape=}"
    )
    assert A.device == B.device, (
        f"Inputs must be on the same device, got {A.device=} and {B.device=}"
    )
    assert A.dtype == B.dtype, (
        f"Inputs must have the same dtype, got {A.dtype=} and {B.dtype=}"
    )
    assert A.is_contiguous() & B.is_contiguous(), (
        f"Inputs must be contiguous, got {A.is_contiguous=} and {B.is_contiguous=}"
    )
    assert A.numel() == B.numel(), (
        f"Inputs must have the same number of elements, got {A.numel()=} and {B.numel()=}"
    )


@triton.jit
def vec_add_fwd_kernel(
    X_ptr,
    Y_ptr,
    Z_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    X = tl.load(X_ptr + offs, mask=mask, other=0.0)
    Y = tl.load(Y_ptr + offs, mask=mask, other=0.0)

    Z = X + Y

    tl.store(pointer=Z_ptr + offs, value=Z, mask=mask)


def vec_add_triton_v1(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    _check_vec_add_inputs(A, B)
    assert A.ndim == B.ndim == 1, (
        f"This kernel doesn't support batched addition, got {A.ndim=} and {B.ndim=}"
    )

    C = torch.empty_like(A)
    n_cols = C.numel()

    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_cols, meta["BLOCK_SIZE"]),)
    vec_add_fwd_kernel[grid](
        A,
        B,
        C,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return C
