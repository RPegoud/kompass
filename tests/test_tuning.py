"""Saving the configs `tune` finds and applying them in `bench`."""

import json

from kompass.harness import tuning
from kompass.harness.schema import Case

from .conftest import FAKE_GPU

PARAMS = {"M": 1024, "N": 1024, "K": 1024}
OPTIONS = {"input_precision": "ieee"}


def row(block: int, ms: float, stage: int = 2, status: str = "ok") -> dict:
    """An un-flattened `run_tune` row, as `tuning.save` receives it."""
    return {
        "impl": "matmul_triton_tiled",
        "dtype": "torch.float32",
        "options": OPTIONS,
        "params": PARAMS,
        "tuning": {"BLOCK_SIZE_M": block},
        "times": [ms * 0.9, ms, ms * 1.1],
        "quantiles": [0.2, 0.5, 0.8],
        "stage": stage,
        "status": status,
    }


def case(**fields) -> Case:
    base = {
        "op": "matmul",
        "impl": "matmul_triton_tiled",
        "params": PARAMS,
        "dtype": "float32",
        "seed": 0,
        "quantiles": (0.2, 0.5, 0.8),
        "options": OPTIONS,
    }
    return Case(**(base | fields))  # type: ignore[arg-type]


def test_save_keeps_the_fastest_stage_two_config(tmp_repo):
    rows = [
        row(64, ms=2.0),
        row(128, ms=1.0),
        row(256, ms=0.5, stage=1),  # stage 1 times are too short to trust
        row(32, ms=0.1, status="incorrect"),
    ]
    [path] = tuning.save(rows, FAKE_GPU, run_id="r1", git_commit="abc")

    [entry] = json.loads(path.read_text())
    assert path.parent.name == FAKE_GPU.replace(" ", "_")
    assert entry["tuning"] == {"BLOCK_SIZE_M": 128}
    assert entry["dtype"] == "float32"  # `torch.` prefix dropped
    assert entry["median_ms"] == 1.0 and entry["run_id"] == "r1"


def test_save_merges_with_existing_entries(tmp_repo):
    tuning.save([row(64, ms=1.0)], FAKE_GPU, "r1", None)
    other_shape = row(128, ms=1.0) | {"params": {"M": 64, "N": 64, "K": 64}}
    tuning.save([other_shape], FAKE_GPU, "r2", None)
    tuning.save([row(32, ms=3.0)], FAKE_GPU, "r3", None)  # re-tuned: replaces r1

    entries = tuning.load(FAKE_GPU, "matmul_triton_tiled")
    assert {e["run_id"] for e in entries.values()} == {"r2", "r3"}


def test_apply_fills_exact_matches_only(tmp_repo):
    tuning.save([row(128, ms=1.0)], FAKE_GPU, "r1", None)
    tuned, other_shape, other_gpu, preset = (
        *tuning.apply([case(), case(params={"M": 64, "N": 64, "K": 64})], FAKE_GPU),
        *tuning.apply([case()], "Another GPU"),
        *tuning.apply([case(tuning={"BLOCK_SIZE_M": 16})], FAKE_GPU),
    )
    assert tuned.tuning == {"BLOCK_SIZE_M": 128}
    assert other_shape.tuning == {}
    assert other_gpu.tuning == {}
    assert preset.tuning == {"BLOCK_SIZE_M": 16}  # explicit tuning wins
