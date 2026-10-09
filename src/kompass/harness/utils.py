import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch
import typer

from .ops import Op
from .spec import KernelSpec


def describe_env() -> dict[str, str | int | None]:
    import torch

    def _sh(*argv) -> str:
        r = subprocess.run(argv, check=False, capture_output=True, text=True)
        return (
            r.stdout.strip()
            if r.returncode == 0
            else f"FAILED({r.returncode}): {r.stderr.strip()}"
        )

    m = re.search(r"release (\d+\.\d+), V(\S+)", _sh("nvcc", "--version"))
    nvcc = m.group(2) if m else "unknown"

    return {
        "nvidia_smi": _sh(
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ),
        "nvcc_version": nvcc,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "device_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch_cuda_version": torch.version.cuda,
    }


def resolve_impl(name: str) -> tuple[Op, KernelSpec]:
    from ..kernels import REGISTRY

    op_name, _, impl_name = name.partition("/")
    impl_name = f"{op_name}_{impl_name}"
    if not impl_name:
        raise typer.BadParameter(f"Expected <op>/<impl>, got {name!r}")
    try:
        entry = REGISTRY[op_name]
    except KeyError:
        raise typer.BadParameter(
            f"Unknown Op {op_name!r} expected one of {', '.join(sorted(REGISTRY))}"
        ) from None
    try:
        return entry.op, entry.impls[impl_name]
    except KeyError:
        raise typer.BadParameter(
            f"Unknown Impl {impl_name!r} for Op {op_name!r}; "
            f"Known: {', '.join(sorted(entry.impls))}"
        ) from None


def resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype), (
            f"Expected a valid torch dtype, got {dtype}"
        )
    return dtype


def repo_root() -> Path:
    for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("repo root not found")


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return None


def parse_params(params: list[str], expected: tuple[str, ...]) -> dict[str, int]:
    """
    Parses a list of params with format `param=value` from the CLI.
    """
    resolved = {param: int(val) for (param, val) in [p.split("=", 1) for p in params]}
    assert all(n in expected for n in resolved)
    return resolved


def get_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")