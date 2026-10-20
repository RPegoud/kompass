"""Report generation: reads benchmark JSONL, writes plots + summary.

Pure data -> plots. No torch, no CUDA, no Modal, no browser: runs on a laptop.
Plots are rendered by xy's native exporter to PNG (embedded in summary.md)
and gathered into a single multi-page vector PDF (report.pdf).
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import xy
from pypdf import PdfWriter

# Vendor spec sheets. Dashed reference lines only -- never reachable in practice.
# H100 SXM5: HBM3 3.35 TB/s; dense (non-sparse) tensor-core TFLOPS.
# fp32 is the non-tensor CUDA-core number.
VENDOR_PEAKS: dict[str, dict[str, float]] = {
    "NVIDIA H100 80GB HBM3": {
        "bandwidth_gbps": 3350.0,
        "tflops_torch.float32": 67.0,  # CUDA cores, no tensor cores
        "tflops_tf32": 494.5,  # tensor cores, tf32
        "tflops_torch.bfloat16": 989.4,  # tensor cores, bf16 dense
    },
}


@dataclass(frozen=True)
class Roofs:
    """Ceilings for one (gpu, dtype) pair, in native units."""

    memory_gbps: float | None
    compute_tflops: float | None  # empirical: best observed torch impl
    vendor_bandwidth_gbps: float | None
    vendor_tflops: float | None


def load_records(
    logs_dir: Path,
    cfg_name: str | None = None,
    run_id: str | None = None,
    gpu: str | None = None,
    latest_only: bool = True,
) -> pd.DataFrame:
    """Load every times.jsonl under logs_dir into one frame, then filter.

    Rows are self-describing, so the directory layout is irrelevant here --
    we glob everything and filter on fields.
    """
    rows: list[dict[str, Any]] = []
    for f in sorted(logs_dir.glob("*/times.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise FileNotFoundError(f"no records found under {logs_dir}")

    df = pd.DataFrame(rows)

    if cfg_name is not None:
        df = df[df["cfg_name"] == cfg_name]
    if run_id is not None:
        df = df[df["run_id"] == run_id]
    if gpu is not None:
        df = df[df["device_name"] == gpu]

    if df.empty:
        raise ValueError("no records left after filtering")

    # Mixing GPUs in one plot is meaningless; make the caller choose.
    gpus = df["device_name"].unique()
    if len(gpus) > 1 and gpu is None:
        raise ValueError(f"records span multiple GPUs {list(gpus)}; pass --gpu")

    if latest_only and run_id is None:
        # Keep the newest run per cfg_name. Re-runs of the same sweep otherwise
        # stack on top of each other in the plots.
        newest = df.groupby("cfg_name")["run_id"].transform(
            lambda s: s.iloc[-1]  # rows arrive in file order == chronological
        )
        df = df[df["run_id"] == newest]

    return df.reset_index(drop=True)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the plots actually use.

    `times` is [p20, p50, p80] in milliseconds (do_bench quantile order).
    """
    df = df.copy()

    ok = df["status"] == "ok"
    times = df.loc[ok, "times"]

    df.loc[ok, "p20_ms"] = times.map(lambda t: t[0])  # type: ignore
    df.loc[ok, "p50_ms"] = times.map(lambda t: t[1])  # type: ignore
    df.loc[ok, "p80_ms"] = times.map(lambda t: t[2])  # type: ignore

    secs = df["p50_ms"] * 1e-3
    df["tflops"] = df["flops"] / secs / 1e12
    df["bandwidth_gbps"] = df["bytes_moved"] / secs / 1e9

    # Algorithmic arithmetic intensity: FLOPs per *ideal* byte moved.
    # bytes_moved is the minimum traffic (inputs + output, counted once), not
    # measured DRAM traffic -- a tiled matmul re-reads tiles. Label plots
    # accordingly.
    df["arithmetic_intensity"] = df["flops"] / df["bytes_moved"]

    param_cols = [c for c in df.columns if c.startswith("param_")]
    df["size_label"] = df[param_cols].apply(
        lambda r: "x".join(str(int(v)) for v in r), axis=1
    )
    df["problem_size"] = df[param_cols].prod(axis=1)

    return df


def compute_roofs(df: pd.DataFrame, dtype: str, baseline: str) -> Roofs:
    """Ceilings for one dtype.

    Compute roof is the best observed baseline (cuBLAS via torch) throughput:
    a measured, in-harness number on the same GPU, which beats a synthetic
    FMA microbenchmark. Memory roof is the measured copy bandwidth.
    """
    sub = df[(df["dtype"] == dtype) & (df["status"] == "ok")]
    if sub.empty:
        return Roofs(None, None, None, None)

    copy_bw = sub["measured_copy_bandwidth_gbps_triton"].iloc[0]
    memory_gbps = copy_bw.get(dtype) if isinstance(copy_bw, dict) else None

    base = sub[sub["impl"] == baseline]
    compute_tflops = float(base["tflops"].max()) if not base.empty else None

    gpu = sub["device_name"].iloc[0]
    peaks = VENDOR_PEAKS.get(gpu, {})

    return Roofs(
        memory_gbps=memory_gbps,
        compute_tflops=compute_tflops,
        vendor_bandwidth_gbps=peaks.get("bandwidth_gbps"),
        vendor_tflops=peaks.get(f"tflops_{dtype}"),
    )


# Fixed categorical order (validated light-surface palette); assigned to impls
# in sorted order so an impl keeps its color across every plot of a report.
PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
SYMBOLS = ["circle", "square", "diamond", "triangle", "hexagon", "pentagon", "star"]
INK = "#52514e"  # reference lines / roofline: text ink, never a series color

WIDTH, HEIGHT = 1000, 520


@dataclass(frozen=True)
class Style:
    color: str
    symbol: str


def impl_styles(df: pd.DataFrame) -> dict[str, Style]:
    impls = sorted(df["impl"].unique())
    return {
        impl: Style(PALETTE[i % len(PALETTE)], SYMBOLS[i % len(SYMBOLS)])
        for i, impl in enumerate(impls)
    }


def resolve_baseline(df: pd.DataFrame, baseline: str | None) -> str:
    """Pick the impl every other one is compared against.

    Records carry the loader's `fn_name` (e.g. `torch_matmul`), not the
    registry key (`matmul_torch`), so default to the single `torch_*` impl.
    An unknown baseline is an error: silently comparing against nothing is
    what made every `pct_of_baseline` NaN.
    """
    impls = sorted(df["impl"].unique())
    if baseline is None:
        torch_impls = [i for i in impls if i.startswith("torch_")]
        if len(torch_impls) != 1:
            raise ValueError(
                f"cannot infer baseline from impls {impls}; pass --baseline"
            )
        return torch_impls[0]
    if baseline not in impls:
        raise ValueError(f"baseline {baseline!r} not in records; impls: {impls}")
    return baseline


def _size_axis(sub: pd.DataFrame) -> tuple[dict[str, int], Any]:
    """Ordinal x positions for problem sizes, ordered by size (not by string),
    plus the matching x axis. Ordinal rather than categorical so every series
    shares one order even when an impl is missing a size.

    Ticks show only the params that vary (e.g. `N` in a K-sweep); the fixed
    ones go in the axis title.
    """
    params = [c for c in sub.columns if c.startswith("param_") and sub[c].notna().any()]
    varying = [c for c in params if sub[c].nunique() > 1] or params
    fixed = [c for c in params if c not in varying]

    sizes = sub.drop_duplicates("size_label").sort_values(
        ["problem_size", "size_label"]
    )
    pos = {label: i for i, label in enumerate(sizes["size_label"])}
    ticks = ["x".join(str(int(r[c])) for c in varying) for _, r in sizes.iterrows()]

    label = "x".join(c.removeprefix("param_") for c in varying)
    if fixed:
        label += (
            " ("
            + ", ".join(
                f"{c.removeprefix('param_')}={int(sub[c].iloc[0])}" for c in fixed
            )
            + ")"
        )
    rotate = sum(map(len, ticks)) > 60
    axis = xy.x_axis(
        label=label,
        label_offset=8 * max(map(len, ticks)) // 2 + 16 if rotate else None,
        tick_values=list(range(len(ticks))),
        tick_labels=ticks,
        tick_label_angle=-30 if rotate else None,
        domain=(-0.4, len(ticks) - 0.6),
    )
    return pos, axis


def _log_axis(axis: Any, values: Any, label: str) -> Any:
    """Log axis with 1-2-5 ticks over the data range (xy labels only decades
    by default, which leaves a 1.5-decade roofline with one tick)."""
    v = pd.Series(values).dropna()
    v = v[v > 0]
    lo, hi = float(v.min()) / 1.5, float(v.max()) * 1.5
    mantissas = (1, 2, 3, 5) if hi / lo < 30 else (1, 2, 5)
    ticks = [
        m * 10.0**e
        for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
        for m in mantissas
        if lo <= m * 10.0**e <= hi
    ]
    if len(ticks) > 10:  # many decades: decades only
        ticks = [t for t in ticks if f"{t:.0e}".startswith("1")]
    return axis(
        label=label,
        type_="log",
        domain=(lo, hi),
        tick_values=ticks,
        tick_labels=[f"{t:g}" for t in ticks],
    )


def _zero_based(*values: Any) -> tuple[float, float]:
    """y domain from 0 to just above the largest data point or reference line."""
    top = max(float(pd.Series(v).max()) for v in values if v is not None)
    return 0.0, top * 1.08


def _series(x: Any, y: Any, name: str, style: Style, lines: bool = True) -> list[Any]:
    marks = [
        xy.scatter(
            list(x),
            list(y),
            color=style.color,
            symbol=style.symbol,
            size=8,
            opacity=1.0,
            name=None if lines else name,
        )
    ]
    if lines:
        marks.insert(
            0, xy.line(list(x), list(y), name=name, color=style.color, width=2)
        )
    return marks


def plot_throughput(
    df: pd.DataFrame, dtype: str, roofs: Roofs, styles: dict[str, Style]
) -> xy.Chart:
    """TFLOPS vs problem size, with ceilings. The headline plot."""
    sub = df[(df["dtype"] == dtype) & (df["status"] == "ok")]
    pos, x_axis = _size_axis(sub)
    marks: list[Any] = []

    for impl, g in sub.groupby("impl"):
        g = g.sort_values("problem_size")
        x = g["size_label"].map(pos)
        hi = g["flops"] / (g["p20_ms"] * 1e-3) / 1e12  # p20 time -> upper TFLOPS
        lo = g["flops"] / (g["p80_ms"] * 1e-3) / 1e12
        marks.append(
            xy.errorbar(
                list(x),
                list(g["tflops"]),
                yerr=(list(g["tflops"] - lo), list(hi - g["tflops"])),
                color=styles[str(impl)].color,
                cap_size=0.08,  # data units (ordinal positions), not px
            )
        )
        marks += _series(x, g["tflops"], str(impl), styles[str(impl)])

    if roofs.vendor_tflops:
        marks.append(
            xy.hline(
                roofs.vendor_tflops,
                text=f"vendor peak ({roofs.vendor_tflops:.0f})",
                color=INK,
                width=1,
            )
        )
    if roofs.compute_tflops:
        marks.append(
            xy.hline(
                roofs.compute_tflops,
                text=f"best observed baseline ({roofs.compute_tflops:.0f})",
                color=INK,
                width=1,
                opacity=0.6,
            )
        )

    return xy.line_chart(
        *marks,
        x_axis,
        xy.y_axis(
            label="TFLOPS (p50, whiskers p20-p80)",
            domain=_zero_based(
                sub["flops"] / (sub["p20_ms"] * 1e-3) / 1e12, roofs.vendor_tflops
            ),
        ),
        xy.legend(loc="best"),
        title=f"Throughput — {dtype}",
        width=WIDTH,
        height=HEIGHT,
    )


def plot_pct_of_baseline(
    df: pd.DataFrame, dtype: str, baseline: str, styles: dict[str, Style]
) -> xy.Chart:
    """Each impl as a fraction of the baseline, per size. The scoreboard."""
    sub = df[(df["dtype"] == dtype) & (df["status"] == "ok")]
    pos, x_axis = _size_axis(sub)
    base = sub[sub["impl"] == baseline].set_index("size_label")["tflops"]

    marks: list[Any] = []
    pcts = [100.0]
    for impl, g in sub.groupby("impl"):
        if impl == baseline:
            continue
        g = g.sort_values("problem_size")
        pct = 100 * g["tflops"].to_numpy() / g["size_label"].map(base).to_numpy()
        pcts += list(pct)
        marks += _series(g["size_label"].map(pos), pct, str(impl), styles[str(impl)])

    marks.append(xy.hline(100, text=baseline, color=INK, width=1))
    return xy.line_chart(
        *marks,
        x_axis,
        xy.y_axis(label="% of baseline TFLOPS", domain=_zero_based(pcts)),
        xy.legend(loc="best"),
        title=f"% of {baseline} — {dtype}",
        width=WIDTH,
        height=HEIGHT,
    )


def plot_roofline(
    df: pd.DataFrame, dtype: str, roofs: Roofs, styles: dict[str, Style]
) -> xy.Chart:
    """Classic roofline: algorithmic intensity vs achieved throughput."""
    sub = df[(df["dtype"] == dtype) & (df["status"] == "ok")]
    marks: list[Any] = []
    xs: list[float] = []

    if roofs.memory_gbps and roofs.compute_tflops:
        # Roof is min(peak_compute, bandwidth * intensity).
        lo = max(sub["arithmetic_intensity"].min() * 0.5, 1e-2)
        hi = sub["arithmetic_intensity"].max() * 2
        xs = [lo * (hi / lo) ** (i / 200) for i in range(201)]
        ys = [
            min(roofs.compute_tflops, roofs.memory_gbps * x / 1e3) for x in xs
        ]  # GB/s * FLOP/byte = GFLOPS -> /1e3 = TFLOPS
        marks.append(xy.line(xs, ys, name="roofline (measured)", color=INK, width=2))

    for impl, g in sub.groupby("impl"):
        marks += _series(
            g["arithmetic_intensity"],
            g["tflops"],
            str(impl),
            styles[str(impl)],
            lines=False,
        )

    return xy.scatter_chart(
        *marks,
        _log_axis(
            xy.x_axis,
            [*sub["arithmetic_intensity"], *xs[:1], *xs[-1:]],
            "algorithmic arithmetic intensity (FLOP / ideal byte)",
        ),
        _log_axis(xy.y_axis, [*sub["tflops"], roofs.compute_tflops], "TFLOPS"),
        xy.legend(loc="lower right"),
        title=f"Roofline — {dtype}",
        width=WIDTH,
        height=HEIGHT,
    )


def plot_accuracy_speed(df: pd.DataFrame, styles: dict[str, Style]) -> xy.Chart:
    """Accuracy cost against speed. The part most GEMM posts skip.

    Color/symbol follow the impl as everywhere else; dtype is the fill:
    the first dtype is solid, the others hollow.
    """
    sub = df[df["status"] == "ok"]
    dtypes = sorted(sub["dtype"].unique())
    marks: list[Any] = []
    for (impl, dtype), g in sub.groupby(["impl", "dtype"]):
        style = styles[str(impl)]
        hollow = dtype != dtypes[0]
        marks.append(
            xy.scatter(
                list(g["tflops"]),
                list(g["max_abs_diff"].clip(lower=1e-8)),
                name=f"{impl} / {str(dtype).replace('torch.', '')}",
                color="#ffffff" if hollow else style.color,
                stroke=style.color if hollow else None,
                stroke_width=2 if hollow else 0,
                symbol=style.symbol,
                size=9,
                opacity=1.0,
            )
        )
    return xy.scatter_chart(
        *marks,
        xy.x_axis(label="TFLOPS"),
        _log_axis(
            xy.y_axis,
            sub["max_abs_diff"].clip(lower=1e-8),
            "max abs diff vs fp32 reference",
        ),
        xy.legend(loc="best"),
        title="Accuracy vs throughput",
        width=WIDTH,
        height=HEIGHT,
    )


def summary_table(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """One row per (impl, dtype, size) -- what goes in the writeup."""
    sub = df[df["status"] == "ok"]
    base = (
        sub.loc[sub["impl"] == baseline, ["dtype", "size_label", "tflops"]]
        .drop_duplicates(["dtype", "size_label"])
        .rename(columns={"tflops": "baseline_tflops"})
    )
    out = sub[
        [
            "impl",
            "dtype",
            "size_label",
            "problem_size",
            "p50_ms",
            "tflops",
            "max_abs_diff",
        ]
    ].merge(base, on=["dtype", "size_label"], how="left")
    out["pct_of_baseline"] = 100 * out["tflops"] / out["baseline_tflops"]
    out = out.sort_values(
        ["dtype", "problem_size", "tflops"], ascending=[True, True, False]
    )
    return out.drop(columns=["problem_size", "baseline_tflops"])


def write_plots(figs: list[tuple[str, xy.Chart]], out_dir: Path) -> Path:
    """One PNG per plot (for the markdown) + every plot as a page of one PDF."""
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    pdf = PdfWriter()
    for name, fig in figs:
        fig.write_image(plots_dir / f"{name}.png", scale=2)
        pdf.append(io.BytesIO(fig.to_image("pdf")))

    pdf_path = out_dir / "report.pdf"
    with open(pdf_path, "wb") as f:
        pdf.write(f)
    return pdf_path


def generate_report(
    logs_dir: Path,
    out_dir: Path,
    cfg_name: str | None = None,
    run_id: str | None = None,
    gpu: str | None = None,
    baseline: str | None = None,
) -> Path:
    df = add_derived(load_records(logs_dir, cfg_name, run_id, gpu))
    baseline = resolve_baseline(df, baseline)
    styles = impl_styles(df)
    out_dir.mkdir(parents=True, exist_ok=True)

    figs: list[tuple[str, xy.Chart]] = []
    for dtype in sorted(df["dtype"].unique()):
        roofs = compute_roofs(df, dtype, baseline)
        short = dtype.replace("torch.", "")
        figs += [
            (f"throughput_{short}", plot_throughput(df, dtype, roofs, styles)),
            (
                f"pct_baseline_{short}",
                plot_pct_of_baseline(df, dtype, baseline, styles),
            ),
            (f"roofline_{short}", plot_roofline(df, dtype, roofs, styles)),
        ]
    figs.append(("accuracy_speed", plot_accuracy_speed(df, styles)))

    write_plots(figs, out_dir)

    table = summary_table(df, baseline)
    table.to_csv(out_dir / "summary.csv", index=False)

    failed = df[df["status"] != "ok"]
    plots_md = "\n\n".join(f"![{name}](plots/{name}.png)" for name, _ in figs)
    summary = out_dir / "summary.md"
    summary.write_text(
        f"# {cfg_name or 'all'} — {df['device_name'].iloc[0]}\n\n"
        f"run_id: {df['run_id'].iloc[0]}  \n"
        f"git: {df.get('git_commit', pd.Series(['?'])).iloc[0]}  \n"
        f"baseline: {baseline}  \n"
        f"cases: {len(df)} ({len(failed)} not ok)  \n"
        "plots: [report.pdf](report.pdf)\n\n"
        "## Results\n\n"
        + table.to_markdown(index=False, floatfmt=".4g")
        + "\n\n## Plots\n\n"
        + plots_md
        + (
            "\n\n## Not ok\n\n"
            + failed[["impl", "dtype", "size_label", "status", "error"]].to_markdown(
                index=False
            )
            if not failed.empty
            else ""
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
