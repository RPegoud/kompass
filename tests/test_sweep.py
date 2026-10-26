"""Expanding a sweep YAML into cases (`kompass.cli.main._build_run`)."""

from collections import Counter

import pytest
import typer

from kompass.cli.main import _build_run


def test_options_expand_on_float32_only():
    """`input_precision` changes the arithmetic on float32 inputs only, so
    bf16 runs each impl once while float32 runs it once per precision."""
    cases = _build_run("matmul_square", "cuda:0").payload.cases
    combos = Counter((c.impl, c.dtype, c.options.get("input_precision")) for c in cases)

    for impl in (
        "matmul_triton_tiled",
        "matmul_triton_tiled_l2_swizzle",
        "matmul_torch",
    ):
        assert (impl, "float32", "ieee") in combos
        assert (impl, "float32", "tf32") in combos
        assert (impl, "bfloat16", "tf32") not in combos
    assert len(set(combos.values())) == 1  # every combo runs every shape once


def test_axes_are_a_cartesian_product():
    run = _build_run("softmax_rowlen", "cuda:0")
    shapes = {tuple(sorted(c.params.items())) for c in run.payload.cases}
    assert all(dict(s)["M"] == 4096 for s in shapes)
    assert len(shapes) == len({dict(s)["N"] for s in shapes})


def test_unknown_config_lists_the_known_ones():
    with pytest.raises(typer.BadParameter, match="matmul_square"):
        _build_run("not_a_sweep", "cuda:0")
