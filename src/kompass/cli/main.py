from pathlib import Path
from typing import Annotated, Any, Literal

import typer

app = typer.Typer(
    name="kompass",
    add_completion=True,
    no_args_is_help=True,
)


@app.command(name="list")
def registry():
    """List the registered ops and kernels."""
    from rich.console import Console
    from rich.table import Table

    from ..kernels import REGISTRY

    console = Console()
    table = Table(title="Kompass Registry")
    table.add_column("Ops", style="magenta")
    table.add_column("Impls")
    print(REGISTRY)
    for op_name in REGISTRY:
        for impl in REGISTRY[op_name].impls:
            table.add_row(op_name, impl)

    console.print(table)


@app.command()
def bench(
    sweep_cfg: Annotated[str, typer.Argument(help="Path to cases/*.yaml")],
    backend: Annotated[
        str, typer.Option(help="Either `local` or `modal:gpu`, e.g. `modal:h100`")
    ] = "local",
    device: str = "cuda:0",
):
    """Run one Op against multiple `Cases` ({impl, shape, dtype} tuples)."""
    import json
    import uuid
    from functools import partial
    from itertools import product

    from omegaconf import OmegaConf

    from ..harness.case import Case
    from ..harness.configs import SweepConfig
    from ..harness.ops import Op
    from ..harness.utils import get_timestamp, git_commit, repo_root
    from ..kernels import REGISTRY
    from ..runner import Payload

    def run_local(
        payload: Payload,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str | int | None]]:
        from ..runner import run_bench

        return run_bench(payload)

    def run_modal(
        payload: Payload, gpu: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str | int | None]]:
        import modal

        from ..infra.modal_runner import app, run_modal_bench

        with modal.enable_output(), app.run():
            return run_modal_bench.with_options(gpu=gpu).remote(payload)

    run_id = uuid.uuid4().hex[:8]
    timestamp = get_timestamp()

    output_dir = repo_root() / "logs" / f"{timestamp}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_file = output_dir / "meta.json"
    times_file = output_dir / "times.jsonl"

    raw = OmegaConf.to_container(OmegaConf.load(sweep_cfg), resolve=True)
    if not isinstance(raw, dict):
        raise typer.BadParameter(
            f"{sweep_cfg} must contain a mapping, got {type(raw).__name__}"
        )
    cfg = SweepConfig(run_id=run_id, **{str(k): v for (k, v) in raw.items()})

    axes: dict[str, list] = cfg.axes
    op: Op = REGISTRY[cfg.op].op
    param_names = op.param_names
    resolved_params = {
        param: val for (param, val) in [(p, axes[p]) for p in param_names]
    }
    dtypes = axes["dtypes"]
    grid = list(product(cfg.impls, dtypes, *[resolved_params[p] for p in param_names]))

    cases = [
        Case(
            op=cfg.op,
            impl=impl,
            params={param: int(val) for (param, val) in zip(param_names, params)},
            dtype=dtype,
            seed=cfg.seed,
            quantiles=cfg.quantiles,
        )
        for impl, dtype, *params in grid
    ]

    payload = Payload(
        run_id=run_id,
        op=op.name,
        cases=cases,
        device=device,
    )

    if backend == "local":
        runner = run_local

    elif backend.startswith("modal:"):
        *_, gpu = backend.partition(":")
        gpu = gpu.upper()
        if gpu == "H100":
            gpu += "!"  # `!` prevents automatic upgrades to H200
        runner = partial(run_modal, gpu=gpu)

    else:
        raise typer.BadParameter(
            f"Expected backend format `modal:gpu`, e.g. `modal:h100`, got {backend}"
        )

    meta, times, env = {}, [], {}

    try:
        meta, times, env = runner(payload)
    except Exception as e:
        meta["error"] = repr(e)
        raise
    finally:
        meta |= {
            "kind": "bench",
            "run_id": run_id,
            "timestamp": timestamp,
            "git_commit": git_commit(),
        }
        for row in times:  # flatten params dict to "param_{name}"
            params = row.pop("params")
            for p_name, p_value in params.items():
                row[f"param_{p_name}"] = p_value

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with open(times_file, "w", encoding="utf-8") as f:
            f.writelines(
                json.dumps(row | env | {"run_id": run_id, "kind": "bench"}) + "\n"
                for row in times
            )


@app.command()
def ingest(report_file: Path) -> None:
    """Parse a `.ncu-rep` report into a record row."""

    import json
    from dataclasses import asdict

    import ncu_report

    from ..harness.records import ProfileRecord
    from ..harness.utils import repo_root

    report_dir = report_file.parent
    uid = report_file.parent.stem
    meta_file = report_dir / "meta.json"
    meta = json.load(meta_file.open())
    ts = meta["timestamp"]

    ctx = ncu_report.load_report(str(report_file))  # type: ignore[call-arg]
    rows = []
    for rng in ctx:
        for i, action in enumerate(rng):
            rows.append(
                ProfileRecord(
                    meta=meta,
                    kernel_name=action.name(),
                    action_idx=i,
                    duration_ns=action["gpu__time_duration.sum"].as_uint64(),
                    dram_bytes=action["dram__bytes.sum"].as_uint64(),
                    dram_pct_peak=action[
                        "dram__throughput.avg.pct_of_peak_sustained_elapsed"
                    ].as_double(),
                    sm_pct_peak=action[
                        "sm__throughput.avg.pct_of_peak_sustained_elapsed"
                    ].as_double(),
                )
            )
    records = [asdict(rec) for rec in rows]

    for rec in records:  # flatten meta, and its params dict to "param_{name}" : value
        rec_meta = rec.pop("meta")
        params = rec_meta.pop("params")
        rec |= rec_meta
        rec["kind"] = "profile"
        for p_name, p_value in params.items():
            rec[f"param_{p_name}"] = p_value

    logs_dir = Path(repo_root()) / "logs"
    records_dir = logs_dir / f"{ts}_{uid}"
    records_dir.mkdir(parents=True, exist_ok=True)

    with open(records_dir / "records.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in records)

    with open(records_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


@app.command()
def profile(
    registry_key: Annotated[
        str,
        typer.Argument(
            help="Registry key as a `op/framework_version` partition, e.g. `vec_add/cuda_v1` instead of `vec_add_cuda_v1`"
        ),
    ],
    p: Annotated[
        list[str],
        typer.Option(
            "-p",
            help="Op parameters, repeated option. Example: -p batch=64 -p seqlen=4096",
        ),
    ],
    dtype: Annotated[str, typer.Option("--dtype")] = "float32",
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    seed: Annotated[int, typer.Option("--seed")] = 0,
    warmup: Annotated[int, typer.Option("--warmup")] = 5,
    set_: Annotated[
        Literal["full", "roofline", "detailed"],
        typer.Option("--set", help="The `ncu` set"),
    ] = "full",
):
    """Profile one case under Nsight Compute."""
    import json
    import subprocess
    import sys
    import uuid

    from ..harness.utils import (
        get_timestamp,
        git_commit,
        parse_params,
        repo_root,
        resolve_impl,
    )

    op, impl = resolve_impl(registry_key)
    params = parse_params(p, expected=impl.op.param_names)
    ts = get_timestamp()

    run_dir = (
        repo_root() / "reports" / registry_key.replace("/", "_") / uuid.uuid4().hex[:8]
    )
    run_dir.mkdir(parents=True)
    meta_file = run_dir / "meta.json"
    meta_file.write_text(
        json.dumps(
            {
                "op": op,
                "impl": impl,
                "params": params,
                "dtype": dtype,
                "device": device,
                "seed": seed,
                "warmup": warmup,
                "ncu_set": set_,
                "git_commit": git_commit(),
                "timestamp": ts,
            },
            indent=2,
        )
    )

    cmd = [
        "ncu",
        "--set",
        set_,
        "--profile-from-start",
        "off",
        "-f",
        "-o",
        str(run_dir / "report"),
        "--",
        sys.executable,
        "-m",
        "kompass.harness.profile",
        str(meta_file),
    ]
    rc = subprocess.run(cmd, check=False).returncode
    if rc == 0:
        ingest(run_dir / "report.ncu-rep")
    raise typer.Exit(rc)


if __name__ == "__main__":
    app()
