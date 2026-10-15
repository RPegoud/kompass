import torch
import triton
import triton.language as tl


def _check_vec_add_inputs(X: torch.Tensor, Y: torch.Tensor) -> None:
    assert X.shape[1] == Y.shape[0], (
        f"X's column dimension must match Y's row dimension, got {X.shape=}{Y.shape=}"
    )
    assert X.device == Y.device, (
        f"Inputs must be on the same device, got {X.device=} and {Y.device=}"
    )
    assert X.dtype == Y.dtype, (
        f"Inputs must have the same dtype, got {X.dtype=} and {Y.dtype=}"
    )
    assert X.is_contiguous() & Y.is_contiguous(), (
        f"Inputs must be contiguous, got {X.is_contiguous=} and {Y.is_contiguous=}"
    )


@triton.jit
def matmul_tiled_fwd(
    X_ptr,
    X_m_stride,
    X_k_stride,
    Y_ptr,
    Y_k_stride,
    Y_n_stride,
    Z_ptr,
    Z_m_stride,
    Z_n_stride,
    M,
    K,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    m_idx = tl.program_id(axis=0)
    n_idx = tl.program_id(axis=1)

    x_desc = tl.make_tensor_descriptor(
        base=X_ptr,
        shape=[M, K],
        strides=[X_m_stride, X_k_stride],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    y_desc = tl.make_tensor_descriptor(
        base=Y_ptr,
        shape=[K, N],
        strides=[Y_k_stride, Y_n_stride],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_M],
    )
    z_desc = tl.make_tensor_descriptor(
        base=Z_ptr,
        shape=[M, N],
        strides=[Z_m_stride, Z_n_stride],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_M],
    )

    z_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)

    m_offset = m_idx * BLOCK_SIZE_M
    n_offset = n_idx * BLOCK_SIZE_N

    for k_offset in range(0, K, BLOCK_SIZE_K):
        x = x_desc.load([m_offset, k_offset])
        y = y_desc.load([k_offset, n_offset])
        z_acc += tl.dot(x, y)

    z_desc.store([m_offset, n_offset], z_acc)


def triton_matmul_tiled(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    _check_vec_add_inputs(X, Y)
    M, K = X.shape
    _, N = Y.shape

    Z = torch.empty((M, N), device="cuda", dtype=X.dtype)

    X_m_stride, X_k_stride = X.stride()
    Y_k_stride, Y_n_stride = Y.stride()
    Z_m_stride, Z_n_stride = Z.stride()

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64

    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_SIZE_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_SIZE_N)

    matmul_tiled_fwd[(NUM_BLOCKS_M, NUM_BLOCKS_N)](
        X_ptr=X,
        X_m_stride=X_m_stride,
        X_k_stride=X_k_stride,
        Y_ptr=Y,
        Y_k_stride=Y_k_stride,
        Y_n_stride=Y_n_stride,
        Z_ptr=Z,
        Z_m_stride=Z_m_stride,
        Z_n_stride=Z_n_stride,
        M=M,
        K=K,
        N=N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,  # type: ignore
        BLOCK_SIZE_N=BLOCK_SIZE_N,  # type: ignore
        BLOCK_SIZE_K=BLOCK_SIZE_K,  # type: ignore
    )

    return Z
