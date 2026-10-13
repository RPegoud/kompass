import torch
import triton
import triton.language as tl


def _check_softmax_inputs(X: torch.Tensor) -> None:
    assert X.is_contiguous(), "The inputs must be contiguous."


@triton.jit
def softmax_triton_single_block_fwd(
    X_ptr,
    X_row_stride,
    Y_ptr,
    Y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    X_ptr += row_id * X_row_stride
    Y_ptr += row_id * Y_row_stride

    X = tl.load(X_ptr + offs, mask=mask, other=-float("inf"))

    m = tl.max(X, axis=0)
    e = tl.exp(X - m)
    d = tl.sum(e, axis=0)
    y = e / d

    tl.store(Y_ptr + offs, y, mask=mask)


@triton.jit
def softmax_triton_online_fwd(
    X_ptr,
    X_row_stride,
    Y_ptr,
    Y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    X_ptr += row_id * X_row_stride
    Y_ptr += row_id * Y_row_stride

    d = 0.0
    m_tm1, m_t = -float("inf"), -float("inf")

    for it in range(0, n_cols, BLOCK_SIZE):
        idx = it + offs
        mask = idx < n_cols
        X_block = tl.load(X_ptr + idx, mask=mask, other=-float("inf"))

        m_block = tl.max(X_block, axis=0)
        m_t = tl.maximum(m_block, m_tm1)
        d = d * tl.exp(m_tm1 - m_t) + tl.sum(tl.exp(X_block - m_t), axis=0)
        m_tm1 = m_t

    for it in range(0, n_cols, BLOCK_SIZE):
        idx = it + offs
        mask = idx < n_cols
        X_block = tl.load(X_ptr + idx, mask=mask, other=0.0)

        Y_block = tl.exp(X_block - m_t) / d
        tl.store(pointer=Y_ptr + idx, value=Y_block, mask=mask)


def _softmax_triton(
    X: torch.Tensor, kernel_fn: triton.JITFunction, BLOCK_SIZE: int
) -> torch.Tensor:
    _check_softmax_inputs(X)
    rows, cols = X.shape

    Y = torch.empty_like(X)

    kernel_fn[(rows,)](
        X_ptr=X,
        X_row_stride=X.stride(0),
        Y_ptr=Y,
        Y_row_stride=Y.stride(0),
        n_cols=cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return Y


def softmax_triton_single_block(X: torch.Tensor) -> torch.Tensor:
    BLOCK_SIZE = triton.next_power_of_2(X.shape[1])
    return _softmax_triton(
        X=X,
        kernel_fn=softmax_triton_single_block_fwd,
        BLOCK_SIZE=int(BLOCK_SIZE),
    )


def softmax_triton_online(
    X: torch.Tensor,
) -> torch.Tensor:  # TODO: add BLOCK_SIZE option
    BLOCK_SIZE = 1024
    return _softmax_triton(
        X=X,
        kernel_fn=softmax_triton_online_fwd,
        BLOCK_SIZE=int(BLOCK_SIZE),
    )
