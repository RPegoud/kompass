from typing import Annotated, Any

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
    table.add_column("Kernels", style="magenta")
    for op_name, impls in REGISTRY.items():
        table.add_row(op_name, str(impls))

    console.print(table)


@app.command()
def profile(
    kernel: Annotated[str, typer.Argument(help="Registry key")],
    p: Annotated[
        list[str],
        typer.Option(
            help="Op parameters, repeated option. Example: -p batch=64 -p seqlen=4096"
        ),
    ],
    dtype: str = "float32",
    device: str = "cuda:0",
    seed: int = 0,
    warmup: int = 5,
):
    """Run one kernel once, scoped for `ncu` capture."""
    import torch

    from ..harness.profile import profile_one
    from ..harness.utils import resolve_dtype, resolve_impl

    _, spec = resolve_impl(kernel)
    expected_params = set(spec.op.param_names)

    resolved_dtype: torch.dtype = resolve_dtype(dtype)
    resolved_device = torch.device(device)
    resolved_params = {param: int(val) for (param, val) in [x.split("=") for x in p]}

    resolved_params_keys = set(resolved_params.keys())
    if resolved_params_keys != expected_params:
        raise typer.BadParameter(
            f"Expected params: {expected_params}, got {resolved_params_keys}, missing {expected_params - resolved_params_keys}"
        )

    # TODO: handle output?
    profile_one(
        spec=spec,
        params=resolved_params,
        dtype=resolved_dtype,
        seed=seed,
        device=resolved_device,
        warmup=warmup,
    )


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
    from datetime import UTC, datetime
    from functools import partial
    from itertools import product

    from omegaconf import OmegaConf

    from ..harness.case import Case
    from ..harness.configs import SweepConfig
    from ..harness.ops import Op
    from ..harness.utils import git_commit, repo_root
    from ..kernels import REGISTRY
    from ..runner import Payload

    def run_local(payload: Payload) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        from ..runner import run_bench

        return run_bench(payload)

    def run_modal(
        payload: Payload, gpu: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        import modal

        from ..infra.modal_runner import app, run_modal_bench

        with modal.enable_output(), app.run():
            return run_modal_bench.with_options(gpu=gpu).remote(payload)

    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")

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

    meta, times = {}, []

    try:
        meta, times = runner(payload)
    except Exception as e:
        meta["error"] = repr(e)
        raise
    finally:
        meta |= {
            "run_id": run_id,
            "timestamp": timestamp,
            "git_commit": git_commit(),
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with open(times_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row | {"run_id": run_id}) + "\n" for row in times)


if __name__ == "__main__":
    app()
