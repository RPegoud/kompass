"""`run_tune`'s search and `run_bench`'s tuning modes, on a fake GPU."""

from dataclasses import replace
from itertools import product

from kompass import runner
from kompass.harness import tuning
from kompass.harness.schema import Case, Payload

from .conftest import FAKE_GPU


def grid(m: int, values: list[int]) -> list[Case]:
    base = Case(
        op="matmul",
        impl="matmul_triton_tiled",
        params={"M": m, "N": 256, "K": 256},
        dtype="bfloat16",
        seed=0,
        quantiles=(0.2, 0.5, 0.8),
    )
    return [
        replace(
            base,
            tuning={"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}
            | {"num_warps": w, "num_stages": 2},
        )
        for bm, w in product(values, (4, 8))
    ]


def test_run_tune_keeps_the_top_k_per_shape(fake_gpu):
    cases = grid(256, [64, 128, 256]) + grid(512, [64, 128, 256])
    result = runner.run_tune(
        Payload(run_id="t", op="matmul", cases=cases, device="cpu"),
        top_k=2,
        slow_factor=1e9,  # time everything in stage 1
    )
    stage2 = [r for r in result.case_results if r["stage"] == 2]

    assert sum(r["stage"] == 1 for r in result.case_results) == len(cases)
    for m in (256, 512):
        kept = {
            (r["tuning"]["BLOCK_SIZE_M"], r["tuning"]["num_warps"])
            for r in stage2
            if r["params"]["M"] == m
        }
        assert kept == {(64, 4), (64, 8)}  # the two fastest fake configs
    assert {rep for _, rep, _ in fake_gpu.calls} == {25, 100}


def test_run_tune_skips_configs_far_slower_than_the_best(fake_gpu):
    """Configs come smallest first, so later ones are slower than the best so
    far; `slow_floor_ms=0` lets the fake millisecond timings be aborted."""
    cases = grid(256, [64, 128, 256])
    result = runner.run_tune(
        Payload(run_id="t", op="matmul", cases=cases, device="cpu"),
        slow_factor=1.0,
        slow_floor_ms=0.0,
    )
    statuses = [r["status"] for r in result.case_results if r["stage"] == 1]

    assert statuses[0] == "ok"  # nothing to compare the first config to
    assert "slow" in statuses
    assert all(r["status"] == "ok" for r in result.case_results if r["stage"] == 2)


def test_run_tune_records_why_configs_were_pruned(fake_gpu):
    huge = grid(256, [64])[0]
    huge = replace(huge, tuning=huge.tuning | {"BLOCK_SIZE_M": 1024})  # > M
    result = runner.run_tune(
        Payload(run_id="t", op="matmul", cases=[huge], device="cpu")
    )
    [pruned] = result.meta["pruned"]
    assert pruned["tuning"]["BLOCK_SIZE_M"] == 1024 and pruned["reason"]
    assert result.case_results == []


def test_run_bench_both_adds_a_tuned_twin_per_saved_config(tmp_repo, fake_gpu):
    tuned_case, untuned_case = grid(256, [64])[0], grid(512, [64])[0]
    tuning.save(
        [
            {
                "impl": tuned_case.impl,
                "dtype": tuned_case.dtype,
                "options": {},
                "params": tuned_case.params,
                "tuning": tuned_case.tuning,
                "times": [1.0, 1.0, 1.0],
                "quantiles": [0.2, 0.5, 0.8],
                "stage": 2,
                "status": "ok",
            }
        ],
        FAKE_GPU,
        "r1",
        None,
    )
    defaults = [replace(c, tuning={}) for c in (tuned_case, untuned_case)]

    def run(mode: str) -> list[dict]:
        payload = Payload("b", "matmul", defaults, "cpu", tuning_mode=mode)  # type: ignore[arg-type]
        return runner.run_bench(payload).case_results

    assert [bool(r["tuning"]) for r in run("untuned")] == [False, False]
    assert [bool(r["tuning"]) for r in run("tuned")] == [True, False]
    assert [bool(r["tuning"]) for r in run("both")] == [False, False, True]
