from typing import Literal

import torch
import triton
import triton.language as tl


def _check_matmul_inputs(X: torch.Tensor, Y: torch.Tensor) -> None:
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
    input_precision: tl.constexpr,
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
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    z_desc = tl.make_tensor_descriptor(
        base=Z_ptr,
        shape=[M, N],
        strides=[Z_m_stride, Z_n_stride],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    z_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    m_offset = m_idx * BLOCK_SIZE_M
    n_offset = n_idx * BLOCK_SIZE_N

    for k_offset in range(0, K, BLOCK_SIZE_K):
        x = x_desc.load([m_offset, k_offset])
        y = y_desc.load([k_offset, n_offset])
        z_acc = tl.dot(x, y, acc=z_acc, input_precision=input_precision)

    z_desc.store([m_offset, n_offset], z_acc)


def matmul_triton_tiled(
    X: torch.Tensor,
    Y: torch.Tensor,
    BLOCK_SIZE_M: int = 128,
    BLOCK_SIZE_N: int = 128,
    BLOCK_SIZE_K: int = 64,
    num_stages: int = 3,
    num_warps: int = 8,
    input_precision: Literal["ieee", "tf32"] = "ieee",
) -> torch.Tensor:
    _check_matmul_inputs(X, Y)
    M, K = X.shape
    _, N = Y.shape

    Z = torch.empty((M, N), device="cuda", dtype=X.dtype)

    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_SIZE_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_SIZE_N)

    matmul_tiled_fwd[(NUM_BLOCKS_M, NUM_BLOCKS_N)](
        X_ptr=X,
        X_m_stride=X.stride(0),
        X_k_stride=X.stride(1),
        Y_ptr=Y,
        Y_k_stride=Y.stride(0),
        Y_n_stride=Y.stride(1),
        Z_ptr=Z,
        Z_m_stride=Z.stride(0),
        Z_n_stride=Z.stride(1),
        M=M,
        K=K,
        N=N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,  # type: ignore
        BLOCK_SIZE_N=BLOCK_SIZE_N,  # type: ignore
        BLOCK_SIZE_K=BLOCK_SIZE_K,  # type: ignore
        input_precision=input_precision,  # type: ignore
        num_stages=num_stages,  # type: ignore
        num_warps=num_warps,  # type: ignore
    )

    return Z


# === L2 Cache Swizzle ===


@triton.jit
def get_swizzled_ordering(
    M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
) -> tuple[int, int]:
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@triton.jit
def matmul_tiled_l2_swizzle_fwd(
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
    GROUP_SIZE_M: tl.constexpr,
    input_precision: tl.constexpr,
):
    pid_m, pid_n = get_swizzled_ordering(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)  # type: ignore

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
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    z_desc = tl.make_tensor_descriptor(
        base=Z_ptr,
        shape=[M, N],
        strides=[Z_m_stride, Z_n_stride],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    z_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N

    for k_offset in range(0, K, BLOCK_SIZE_K):
        x = x_desc.load([m_offset, k_offset])
        y = y_desc.load([k_offset, n_offset])
        z_acc = tl.dot(x, y, acc=z_acc, input_precision=input_precision)

    z_desc.store([m_offset, n_offset], z_acc)


def matmul_triton_tiled_l2_swizzle(
    X: torch.Tensor,
    Y: torch.Tensor,
    BLOCK_SIZE_M: int = 128,
    BLOCK_SIZE_N: int = 128,
    BLOCK_SIZE_K: int = 64,
    GROUP_SIZE_M: int = 8,
    num_stages: int = 3,
    num_warps: int = 8,
    input_precision: Literal["ieee", "tf32"] = "ieee",
) -> torch.Tensor:
    _check_matmul_inputs(X, Y)
    M, K = X.shape
    _, N = Y.shape

    Z = torch.empty((M, N), device="cuda", dtype=X.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    matmul_tiled_l2_swizzle_fwd[grid](
        X_ptr=X,
        X_m_stride=X.stride(0),
        X_k_stride=X.stride(1),
        Y_ptr=Y,
        Y_k_stride=Y.stride(0),
        Y_n_stride=Y.stride(1),
        Z_ptr=Z,
        Z_m_stride=Z.stride(0),
        Z_n_stride=Z.stride(1),
        M=M,
        K=K,
        N=N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,  # type: ignore
        BLOCK_SIZE_K=BLOCK_SIZE_K,  # type: ignore
        BLOCK_SIZE_N=BLOCK_SIZE_N,  # type: ignore
        GROUP_SIZE_M=GROUP_SIZE_M,  # type: ignore
        input_precision=input_precision,  # type: ignore
        num_stages=3,  # type: ignore
        num_warps=8,  # type: ignore
    )

    return Z
