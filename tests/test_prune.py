"""The kernels' `prune` hooks: configs rejected before they are compiled."""

import torch

from kompass.kernels.matmul import prune_matmul_config
from kompass.kernels.softmax import prune_softmax_config

from .conftest import SHARED_MEMORY

LIMITS = {"shared_memory_per_block_optin": SHARED_MEMORY}
SQUARE = {"M": 4096, "N": 4096, "K": 4096}


def matmul(**tuning: int) -> dict[str, int]:
    base = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}
    return base | {"num_warps": 8, "num_stages": 3} | tuning


def test_matmul_keeps_a_config_that_fits():
    skip, reason = prune_matmul_config(SQUARE, matmul(), torch.bfloat16, LIMITS)
    assert not skip and reason is None


def test_matmul_rejects_configs_over_shared_memory():
    """4 stages x (256 + 256) x 64 x 4 B = 512 KB, more than an H100 block gets."""
    tuning = matmul(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, num_stages=4)
    skip, reason = prune_matmul_config(SQUARE, tuning, torch.float32, LIMITS)
    assert skip and reason is not None and "shared memory" in reason


def test_matmul_rejects_blocks_larger_than_the_problem():
    params = {"M": 4096, "N": 4096, "K": 32}
    skip, _ = prune_matmul_config(
        params, matmul(BLOCK_SIZE_K=64), torch.bfloat16, LIMITS
    )
    assert skip


def test_softmax_single_block_is_never_pruned():
    skip, _ = prune_softmax_config(
        {"M": 4096, "N": 262144}, {"num_warps": 4}, torch.float32, LIMITS
    )
    assert not skip


def test_softmax_rejects_wasted_blocks():
    params = {"M": 4096, "N": 1024}
    larger_than_row = {"BLOCK_SIZE": 4096, "num_warps": 4, "num_stages": 2}
    idle_threads = {"BLOCK_SIZE": 128, "num_warps": 8, "num_stages": 2}
    fine = {"BLOCK_SIZE": 1024, "num_warps": 8, "num_stages": 2}

    assert prune_softmax_config(params, larger_than_row, torch.float32, LIMITS)[0]
    assert prune_softmax_config(params, idle_threads, torch.float32, LIMITS)[0]
    assert not prune_softmax_config(params, fine, torch.float32, LIMITS)[0]
