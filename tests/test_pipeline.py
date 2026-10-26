"""tune -> saved configs -> bench --tuning both -> report, through the CLI."""

import json

import pandas as pd
from typer.testing import CliRunner

from kompass.cli.main import app
from kompass.harness import tuning
from kompass.harness.report import generate_report

from .conftest import FAKE_GPU

cli = CliRunner()


def test_tune_bench_report(tmp_repo, fake_gpu):
    result = cli.invoke(app, ["tune", "matmul_square"])
    assert result.exit_code == 0, result.output

    saved = sorted((tmp_repo / "tuning" / FAKE_GPU.replace(" ", "_")).glob("*.json"))
    assert [p.stem for p in saved] == [
        "matmul_triton_tiled",
        "matmul_triton_tiled_l2_swizzle",
    ]
    entries = json.loads(saved[0].read_text())
    # one entry per (precision, shape): fp32-ieee, fp32-tf32 and bf16 per shape
    assert len(entries) % 3 == 0
    # the fake GPU rewards the smallest values: the best config is the smallest
    smallest = {k: min(v) for k, v in _space("matmul_triton_tiled").items()}
    assert all(e["tuning"] == smallest for e in entries)

    result = cli.invoke(app, ["bench", "matmul_square", "--tuning", "both"])
    assert result.exit_code == 0, result.output

    summary = generate_report(
        logs_dir=tmp_repo / "logs",
        out_dir=tmp_repo / "reports" / "matmul_square",
        cfg_name="matmul_square",
    )
    table = pd.read_csv(summary.parent / "summary.csv")
    tuned = table.loc[table["impl"].str.endswith(" (tuned)")]
    assert not tuned.empty and bool(tuned["pct_of_untuned"].gt(100).all())
    assert (
        table.loc[~table["impl"].str.endswith(" (tuned)"), "pct_of_untuned"]
        .isna()
        .all()
    )

    plots = {p.stem for p in (summary.parent / "plots").glob("*.png")}
    assert {
        "matmul_square_tuning_gain_float32",
        "matmul_square_roofline_bfloat16",
    } <= plots
    assert (summary.parent / "report.pdf").stat().st_size > 0
    # the tune run shares the logs and cfg_name, but only the bench run is reported
    metas = [
        json.loads((d / "meta.json").read_text()) for d in (tmp_repo / "logs").iterdir()
    ]
    kinds = {m["run_id"]: m["kind"] for m in metas}
    runs_line = next(
        line for line in summary.read_text().splitlines() if line.startswith("runs:")
    )
    for run_id, kind in kinds.items():
        assert (run_id in runs_line) == (kind == "bench")


def _space(impl: str) -> dict[str, list[int]]:
    from kompass.kernels import REGISTRY

    space = REGISTRY["matmul"].impls[impl].search_space
    assert space is not None
    return space


def test_tuning_dir_is_redirected(tmp_repo):
    assert tuning.TUNING_DIR == tmp_repo / "tuning"
