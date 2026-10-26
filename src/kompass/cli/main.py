from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import typer

if TYPE_CHECKING:
    from ..harness.schema import Run

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
    for op_name in REGISTRY:
        for impl in REGISTRY[op_name].impls:
            table.add_row(op_name, impl)
        table.add_section()
    console.print(table)


@app.command()
def runs(
    op: Annotated[
        str | None, typer.Option(help="Op to filter for, e.g. `matmul`")
    ] = None,
    config_name: Annotated[
        str | None,
        typer.Option("--config", help="Config to filter for, e.g. `matmul_k_sweep`"),
    ] = None,
    n: Annotated[int, typer.Option(help="Maximum number of logs to show")] = 20,
):
    """Lists the most recent runs recorded in `logs`."""
    import json
    import os

    from rich.console import Console
    from rich.table import Table

    from ..harness.utils import repo_root

    root = repo_root()
    logs_dir = root / "logs"
    logs = sorted(os.listdir(logs_dir), reverse=True)

    console = Console()
    table = Table(title="Logs")
    table.add_column("Path")
    table.add_column("Timestamp")
    table.add_column("Uid")
    table.add_column("Op")
    table.add_column("Config")

    counter = 0

    for log in logs:
        ts, op_name, uid = log.split("__", 3)
        if op and op_name != op:
            continue

        with open(str(logs_dir / log / "meta.json"), "r") as f:
            meta = json.load(f)
            cfg_name = meta.get("cfg_name", "-")  # profile runs have no sweep config
        if config_name and config_name != cfg_name:
            continue

        table.add_row(log, ts, uid, op_name, cfg_name)
        counter += 1

        if counter == n:
            break
    console.print(table)


def _build_run(config_name: str, device: str) -> Run:
    """Load `cases/<config_name>.yaml`, validate it and expand it into a `Run`."""
    import uuid
    from itertools import product

    from omegaconf import OmegaConf

    from ..harness.ops import Op
    from ..harness.schema import Case, Payload, Run, SweepConfig
    from ..harness.utils import get_timestamp, repo_root, resolve_dtype
    from ..kernels import REGISTRY

    run_id = uuid.uuid4().hex[:8]
    timestamp = get_timestamp()

    cases_dir = Path(__file__).resolve().parents[1] / "cases"
    sweep_path = cases_dir / f"{config_name}.yaml"
    if not sweep_path.is_file():
        available = sorted(p.stem for p in cases_dir.glob("*.yaml"))
        raise typer.BadParameter(
            f"Unknown config `{config_name}`, expected one of {available}"
        )

    raw = OmegaConf.to_container(OmegaConf.load(sweep_path), resolve=True)
    if not isinstance(raw, dict):
        raise typer.BadParameter(
            f"{sweep_path} must contain a mapping, got {type(raw).__name__}"
        )
    cfg = SweepConfig(run_id=run_id, **{str(k): v for (k, v) in raw.items()})
    op: Op = REGISTRY[cfg.op].op

    output_dir = repo_root() / "logs" / f"{timestamp}__{op.name}__{run_id}"

    params_set: list[dict[str, int]]
    if cfg.shapes is not None:
        params_set = cfg.shapes
    elif cfg.axes is not None:
        params_set = [
            dict(zip(cfg.axes.keys(), values)) for values in product(*cfg.axes.values())
        ]
    else:
        raise ValueError("Neither `axes` nor `shapes` is set.")

    param_names = op.param_names
    for ps in params_set:
        if set(ps) != set(param_names):
            raise typer.BadParameter(
                f"{cfg.op} expects {param_names}, got {sorted(ps)}"
            )

    sweep_options = cfg.options or {}
    unknown = set(sweep_options) - set(op.option_names or ())
    if unknown:
        raise typer.BadParameter(
            f"{cfg.op} has options {op.option_names}, got {sorted(unknown)}"
        )

    cases = []
    for impl, dtype in product(cfg.impls, cfg.dtypes):
        spec = REGISTRY[cfg.op].impls[impl]
        names = [n for n in spec.option_names if n in sweep_options]
        combos = [
            dict(zip(names, values))
            for values in product(*(sweep_options[n] for n in names))
        ]
        # Options that don't change the arithmetic on this dtype (e.g. tf32 on
        # bf16 inputs) would rerun the same kernel: keep one per `math`.
        by_math = {}
        for options in combos:
            by_math.setdefault(spec.math(resolve_dtype(dtype), options), options)
        cases += [
            Case(
                op=cfg.op,
                impl=impl,
                params=params,
                dtype=dtype,
                seed=cfg.seed,
                quantiles=cfg.quantiles,
                options=options,
            )
            for options in by_math.values()
            for params in params_set
        ]

    payload = Payload(
        run_id=run_id,
        op=op.name,
        cases=cases,
        device=device,
    )
    return Run(
        config_name=config_name,
        run_id=run_id,
        timestamp=timestamp,
        output_dir=output_dir,
        payload=payload,
    )


@app.command()
def bench(
    config_names: Annotated[
        list[str],
        typer.Argument(
            help="The configuration name(s), as defined in `src/kompass/cases/*.yaml`, e.g. `vec_add_sweep matmul_k_sweep`."
        ),
    ],
    backend: Annotated[
        str, typer.Option(help="Either `local` or `modal:gpu`, e.g. `modal:h100`")
    ] = "local",
    device: str = "cuda:0",
):
    """Run one or more sweeps, each benchmarking one Op against multiple `Cases`
    ({impl, shape, dtype} tuples)."""
    import json
    from collections import Counter

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from ..harness.utils import git_commit
    from ..runner import select_runner

    runs = [_build_run(config_name, device) for config_name in config_names]
    runner = select_runner(backend)

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}", style="dim"),
        BarColumn(bar_width=30, style="dim", complete_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        tasks = {
            run.run_id: progress.add_task(
                f"{run.config_name} ({run.run_id})", total=len(run.payload.cases)
            )
            for run in runs
        }
        results = runner(
            [run.payload for run in runs],
            on_case_done=lambda run_id: progress.advance(tasks[run_id]),
        )

    git = git_commit()
    n_failed = 0
    for run, result in zip(runs, results):
        if isinstance(result, BaseException):
            meta: dict[str, Any] = {"error": repr(result)}
            times: list[dict[str, Any]] = []
            summary = f"failed: {meta['error']}"
            n_failed += 1
        else:
            meta, times = result.meta, result.times
            summary = dict(Counter(r["status"] for r in times))
            for row in times:  # flatten params dict to "param_{name}"
                params = row.pop("params")
                for p_name, p_value in params.items():
                    row[f"param_{p_name}"] = p_value
                row |= result.env
                row |= {
                    "cfg_name": run.config_name,
                    "run_id": run.run_id,
                    "kind": "bench",
                    "git_commit": git,
                    "measured_copy_bandwidth_gbps_triton": meta[
                        "measured_copy_bandwidth_gbps_triton"
                    ],
                }

        meta |= {
            "cfg_name": run.config_name,
            "run_id": run.run_id,
            "timestamp": run.timestamp,
            "kind": "bench",
            "git_commit": git,
        }

        run.output_dir.mkdir(parents=True, exist_ok=True)
        with open(run.output_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with open(run.output_dir / "times.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row) + "\n" for row in times)

        typer.echo(f"{run.config_name} ({run.run_id}): {summary}")

    if n_failed:
        typer.echo(f"{n_failed}/{len(runs)} sweeps failed.", err=True)
        raise typer.Exit(1)
    typer.echo("Run complete!")


@app.command()
def tune(
    config_names: Annotated[
        list[str],
        typer.Argument(
            help="The configuration name(s), as defined in `src/kompass/cases/*.yaml`, e.g. `vec_add_sweep matmul_k_sweep`."
        ),
    ],
    backend: Annotated[
        str, typer.Option(help="Either `local` or `modal:gpu`, e.g. `modal:h100`")
    ] = "local",
    device: str = "cuda:0",
) -> None:
    import json
    from collections import Counter, defaultdict
    from dataclasses import replace
    from itertools import product

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from ..harness.schema import BenchResult, Case, Payload
    from ..harness.utils import git_commit, resolve_dtype
    from ..kernels import REGISTRY
    from ..runner import select_runner

    # 1. build the sweep: `build_runs` + expand with the hparams grid
    # group cases by impl/math/dtype so that one compile handles every case in a group
    runs = [_build_run(config_name, device) for config_name in config_names]
    groups: list[tuple[Run, Payload]] = []
    for run in runs:
        cases: list[Case] = run.payload.cases
        by_group: dict[str, list[Case]] = defaultdict(list)
        for case in cases:
            spec = REGISTRY[case.op].impls[case.impl]
            search_space = spec.search_space

            if search_space is None:  # ignore impls with no search space
                continue

            m = spec.math(resolve_dtype(case.dtype), case.options)
            for param_set in product(*search_space.values()):
                by_group[(case.impl, m, case.dtype)].append(
                    replace(
                        case,
                        tuning={
                            name: value
                            for (name, value) in zip(search_space, param_set)
                        },
                    )
                )
        groups += [(run, replace(run.payload, cases=cs)) for cs in by_group.values()]

    runner = select_runner(backend, kind="tune")

    # 2. coarse run over the whole tuning grid
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}", style="dim"),
        BarColumn(bar_width=30, style="dim", complete_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        tasks = {
            run.run_id: progress.add_task(
                f"{run.config_name} ({run.run_id})", total=len(run.payload.cases)
            )
            for run in runs
        }
        results: list[BenchResult | BaseException] = runner(
            [payload for (_, payload) in groups],
            on_case_done=lambda run_id: progress.advance(tasks[run_id]),
        )

    # TODO: group runs by math, params, impl and filter top 5 tunings

    git = git_commit()
    n_failed = 0
    for run, _, result in zip(groups, results):
        if isinstance(result, BaseException):
            meta: dict[str, Any] = {"error": repr(result)}
            times: list[dict[str, Any]] = []
            summary = f"failed: {meta['error']}"
            n_failed += 1
        else:
            meta, times = result.meta, result.times
            summary = dict(Counter(r["status"] for r in times))
            for row in times:  # flatten params dict to "param_{name}"
                params = row.pop("params")
                for p_name, p_value in params.items():
                    row[f"param_{p_name}"] = p_value
                row |= result.env
                row |= {
                    "cfg_name": run.config_name,
                    "run_id": run.run_id,
                    "kind": "tune",
                    "git_commit": git,
                    "measured_copy_bandwidth_gbps_triton": meta[
                        "measured_copy_bandwidth_gbps_triton"
                    ],
                }

        meta |= {
            "cfg_name": run.config_name,
            "run_id": run.run_id,
            "timestamp": run.timestamp,
            "kind": "bench",
            "git_commit": git,
        }

        run.output_dir.mkdir(parents=True, exist_ok=True)
        with open(run.output_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with open(run.output_dir / "times.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row) + "\n" for row in times)

        typer.echo(f"{run.config_name} ({run.run_id}): {summary}")

    if n_failed:
        typer.echo(f"{n_failed}/{len(runs)} sweeps failed.", err=True)
        raise typer.Exit(1)
    typer.echo("Run complete!")

    # TODO: Persist configs to /tuning
    # TODO: Print a table of configs and best parameters


@app.command()
def ingest(report_file: Path) -> None:
    """Parse a `.ncu-rep` report into a record row."""

    import json
    from dataclasses import asdict

    import ncu_report

    from ..harness.schema import ProfileRecord
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
    records_dir = logs_dir / f"{ts}__{meta['op']}__{uid}"
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
            help="Registry key as a `op/framework_version` partition, e.g. `vec_add/cuda` instead of `vec_add_cuda`"
        ),
    ],
    p: Annotated[
        list[str],
        typer.Option(
            "-p",
            help="Op parameters, repeated option. Example: -p batch=64 -p seqlen=4096",
        ),
    ],
    o: Annotated[
        list[str] | None,
        typer.Option(
            "-o",
            help="Kernel options, repeated option. Example: -o input_precision=tf32",
        ),
    ] = None,
    dtype: Annotated[str, typer.Option("--dtype")] = "float32",
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    seed: Annotated[int, typer.Option("--seed")] = 0,
    warmup: Annotated[int, typer.Option("--warmup")] = 25,
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

    op, spec = resolve_impl(registry_key)
    params = parse_params(p, expected=op.param_names)
    options = dict(kv.split("=", 1) for kv in o or [])
    if unknown := set(options) - set(spec.option_names):
        raise typer.BadParameter(
            f"{registry_key} accepts options {spec.option_names}, got {sorted(unknown)}"
        )
    op_key, _, impl_suffix = registry_key.partition("/")
    ts = get_timestamp()

    run_dir = (
        repo_root() / "reports" / registry_key.replace("/", "_") / uuid.uuid4().hex[:8]
    )
    run_dir.mkdir(parents=True)
    meta_file = run_dir / "meta.json"
    meta_file.write_text(
        json.dumps(
            {
                "op": op_key,
                "impl": impl_suffix,
                "params": params,
                "options": options,
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


@app.command()
def report(
    config_name: Annotated[
        str | None,
        typer.Option(
            "--config",
            help="The configuration name, as defined in `src/kompass/cases/*.yaml`, e.g. `vec_add_sweep`.",
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="The specific `run_id` to filter for."),
    ] = None,
    gpu: Annotated[
        str | None, typer.Option("--gpu", help="The specific `gpu` to filter for.")
    ] = None,
    baseline: Annotated[
        str | None,
        typer.Option(
            "--baseline", help="impl name as recorded; default: the torch_* impl"
        ),
    ] = None,
    out: Annotated[str, typer.Option("--out")] = "reports",
):
    """Generate plots and a summary table from benchmark records."""
    from ..harness.report import generate_report
    from ..harness.utils import repo_root

    root = repo_root()
    summary = generate_report(
        logs_dir=root / "logs",
        out_dir=root / out / (config_name or "all"),
        cfg_name=config_name,
        run_id=run_id,
        gpu=gpu,
        baseline=baseline,
    )
    print(f"wrote {summary} and {summary.parent / 'report.pdf'}")


if __name__ == "__main__":
    app()
