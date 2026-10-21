"""Report generation: reads benchmark JSONL, writes plots + summary.

Pure data -> plots. No torch, no CUDA, no Modal, no browser: runs on a laptop.
Plots are rendered by xy's native exporter to PNG (embedded in summary.md)
and gathered into a single multi-page vector PDF (report.pdf).

Every case is scored against its own ceiling,
    min(compute peak of the arithmetic it runs, copy bandwidth x intensity),
so memory- and compute-bound kernels are judged on one scale.
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

# Dense vendor peaks in TFLOPS, keyed by the record's `math` (KernelSpec.math).
# H100 SXM5: float32 on CUDA cores; tf32 / bfloat16 on tensor cores.
VENDOR_PEAKS: dict[str, dict[str, float]] = {
    "NVIDIA H100 80GB HBM3": {"float32": 67.0, "tf32": 494.5, "bfloat16": 989.4},
}

SURFACE = "#FEFEFE"
GRID = "#EAEAEA"
TEXT = "#000000"
MUTED = "#020202A2"  # reference lines / roofline / baseline: never a palette hue
PALETTE = ["#780116", "#f7b538", "#db7c26", "#d8572a", "#c32f27"]


SYMBOLS = ["circle", "square", "diamond", "triangle", "hexagon", "pentagon", "star"]

WIDTH, HEIGHT = 1000, 520


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
    rows = [
        json.loads(line)
        for f in sorted(logs_dir.glob("*/times.jsonl"))
        for line in f.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
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
    if len(gpus) > 1:
        raise ValueError(f"records span multiple GPUs {list(gpus)}; pass --gpu")

    if latest_only and run_id is None:
        # Keep the newest run per cfg_name. Re-runs of the same sweep otherwise
        # stack on top of each other in the plots.
        newest = df.groupby("cfg_name")["run_id"].transform(
            lambda s: s.iloc[-1]  # rows arrive in file order == chronological
        )
        df = df[df["run_id"] == newest]

    return df.reset_index(drop=True)


def qname(q: float) -> str:
    """0.2 -> 'p20', 0.995 -> 'p99.5'."""
    return f"p{q * 100:g}"


def _time_stats(times: list[float], qs: list[float]) -> dict[str, float]:
    """Headline time = quantile closest to the median; whiskers = the
    outermost two. `times` (ms) is in `quantiles` order (do_bench)."""
    lo, hi = qs.index(min(qs)), qs.index(max(qs))
    center = min(range(len(qs)), key=lambda i: abs(qs[i] - 0.5))
    return {
        "t_ms": times[center],
        "t_lo_ms": times[lo],
        "t_hi_ms": times[hi],
        "q_center": qs[center],
        "q_lo": qs[lo],
        "q_hi": qs[hi],
    }


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Throughputs, sizes and the per-case ceiling, for the ok rows of one op."""
    df = df[df["status"] == "ok"].copy()
    df = df.join(
        pd.DataFrame(
            [_time_stats(t, q) for t, q in zip(df["times"], df["quantiles"])],
            index=df.index,
        )
    )

    secs = df["t_ms"] * 1e-3
    df["tflops"] = df["flops"] / secs / 1e12
    df["gbps"] = df["bytes_moved"] / secs / 1e9
    # FLOPs per *ideal* byte (inputs + output once), not measured DRAM traffic.
    df["arithmetic_intensity"] = df["flops"] / df["bytes_moved"]

    # Memory roof: best measured copy bandwidth (any dtype: bandwidth is a
    # property of the GPU). Compute roof: vendor peak of the arithmetic run.
    df["mem_gbps"] = df["measured_copy_bandwidth_gbps_triton"].map(
        lambda bw: max(bw.values())  # type: ignore
    )
    df["peak_tflops"] = [
        VENDOR_PEAKS[gpu][m] for gpu, m in zip(df["device_name"], df["math"])
    ]
    mem_tflops = df["mem_gbps"] * df["arithmetic_intensity"] / 1e3
    df["bound"] = (mem_tflops < df["peak_tflops"]).map(
        {True: "memory", False: "compute"}
    )
    df["pct_of_ceiling"] = 100 * df["tflops"] / mem_tflops.clip(upper=df["peak_tflops"])

    # Params of *this* op only: frames built from several logs carry other
    # ops' param columns (e.g. matmul's K next to softmax), all NaN here.
    all_params = [c for c in df.columns if c.startswith("param_")]
    params = [c for c in all_params if df[c].notna().all()]
    df = df.drop(columns=[c for c in all_params if c not in params])
    df["size_label"] = df[params].apply(
        lambda r: "x".join(map(str, map(int, r))), axis=1
    )
    df["problem_size"] = df[params].prod(axis=1)
    return df


@dataclass(frozen=True)
class Metric:
    """The throughput an op is judged in: GB/s when most of its cases are
    memory-bound, TFLOPS otherwise."""

    column: str
    unit: str

    @staticmethod
    def for_op(df: pd.DataFrame) -> Metric:
        memory = (df["bound"] == "memory").mean() > 0.5
        return Metric("gbps", "GB/s") if memory else Metric("tflops", "TFLOPS")


def _chrome() -> list[Any]:
    """Theme + legend shared by every chart."""
    return [
        xy.theme(
            background=SURFACE,
            plot_background=SURFACE,
            grid_color=GRID,
            axis_color=GRID,
            text_color=TEXT,
            annotation_text=MUTED,
            legend_bg=SURFACE,
        ),
    ]


@dataclass(frozen=True)
class Style:
    color: str
    symbol: str


def impl_styles(df: pd.DataFrame, baseline: str) -> dict[str, Style]:
    impls = [i for i in sorted(df["impl"].unique()) if i != baseline]
    styles = {
        impl: Style(PALETTE[i % len(PALETTE)], SYMBOLS[i % len(SYMBOLS)])
        for i, impl in enumerate(impls)
    }
    styles[baseline] = Style(MUTED, "circle")
    return styles


def resolve_baseline(df: pd.DataFrame, baseline: str | None) -> str:
    """The impl every other one is compared against, for one op: `--baseline`
    if it belongs to this op, else the op's single torch impl."""
    impls = sorted(df["impl"].unique())
    if baseline in impls:
        return baseline  # type: ignore
    torch_impls = sorted(df.loc[df["framework"] == "torch", "impl"].unique())
    if len(torch_impls) != 1:
        raise ValueError(f"cannot infer baseline from impls {impls}; pass --baseline")
    return torch_impls[0]


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


def quantile_desc(df: pd.DataFrame) -> tuple[str, str]:
    """(headline, whiskers) labels for the quantiles behind `df`'s times,
    e.g. ('p50', 'p20-p80')."""
    sets = df[["q_center", "q_lo", "q_hi"]].drop_duplicates()
    if len(sets) != 1:
        return "central quantile", "outer quantiles"
    c, lo, hi = sets.iloc[0]
    return qname(c), (f"{qname(lo)}-{qname(hi)}" if lo != hi else "")


def _zero_based(*values: Any) -> tuple[float, float]:
    """y domain from 0 to just above the largest data point or reference line."""
    top = max(float(pd.Series(v).max()) for v in values)
    return 0.0, top * 1.08


def _series(
    x: Any, y: Any, name: str, style: Style, lines: bool = True, hollow: Any = None
) -> list[Any]:
    """One impl: a smooth line plus markers; `hollow` marks points to draw
    unfilled (memory-bound cases on the ceiling plot)."""
    x, y = list(x), list(y)
    hollow = list(hollow) if hollow is not None else [False] * len(x)
    marks = [
        xy.scatter(
            [a for a, h in zip(x, hollow) if h == fill],
            [b for b, h in zip(y, hollow) if h == fill],
            color=SURFACE if fill else style.color,
            stroke=style.color if fill else SURFACE,
            stroke_width=2 if fill else 1.5,
            symbol=style.symbol,
            size=9,
            opacity=1.0,
            name=None if lines else name,
        )
        for fill in (False, True)
        if fill in hollow
    ]
    if lines:
        marks.insert(
            0, xy.line(x, y, name=name, color=style.color, width=2.5, curve="smooth")
        )
    return marks


def _line_chart(marks: list[Any], x_axis: Any, y_label: str, y_max: Any, title: str):
    return xy.line_chart(
        *marks,
        *_chrome(),
        x_axis,
        xy.y_axis(label=y_label, domain=_zero_based(*y_max)),
        xy.legend(loc="best"),
        title=title,
        width=WIDTH,
        height=HEIGHT,
    )


def plot_throughput(sub: pd.DataFrame, metric: Metric, styles: dict[str, Style]):
    """Throughput vs problem size, with the ceilings of this unit."""
    pos, x_axis = _size_axis(sub)
    center, whiskers = quantile_desc(sub)
    col = metric.column
    marks: list[Any] = []

    for impl, g in sub.groupby("impl"):
        g = g.sort_values("problem_size")
        x = g["size_label"].map(pos)
        if whiskers:  # throughput ~ 1/time
            hi, lo = (
                g[col] * g["t_ms"] / g["t_lo_ms"],
                g[col] * g["t_ms"] / g["t_hi_ms"],
            )
            marks.append(
                xy.errorbar(
                    list(x),
                    list(g[col]),
                    yerr=(list(g[col] - lo), list(hi - g[col])),
                    color=styles[str(impl)].color,
                    cap_size=0.08,  # data units (ordinal positions), not px
                    opacity=0.7,
                )
            )
        marks += _series(x, g[col], str(impl), styles[str(impl)])

    if metric.column == "gbps":
        roofs = {"measured copy bandwidth": sub["mem_gbps"].max()}
    else:
        roofs = sub.groupby("math")["peak_tflops"].first()
        roofs = {f"{m} peak": v for m, v in roofs.items()}
    for text, v in roofs.items():
        marks.append(xy.hline(v, text=f"{text} ({v:.0f})", color=MUTED, width=1))

    return _line_chart(
        marks,
        x_axis,
        f"{metric.unit} ({center}" + (f", whiskers {whiskers})" if whiskers else ")"),
        [sub[col] * sub["t_ms"] / sub["t_lo_ms"], *roofs.values()],
        f"Throughput — {sub['dtype'].iloc[0]}",
    )


def plot_pct(
    sub: pd.DataFrame,
    column: str,
    ref: str,
    title: str,
    styles: dict[str, Style],
    hollow_memory_bound: bool = False,
):
    """Each impl as a percentage of a reference (the baseline, or its own
    ceiling), per size, with the 100% line labelled `ref`."""
    pos, x_axis = _size_axis(sub)
    marks: list[Any] = []
    for impl, g in sub.dropna(subset=[column]).groupby("impl"):
        g = g.sort_values("problem_size")
        hollow = g["bound"] == "memory" if hollow_memory_bound else None
        marks += _series(
            g["size_label"].map(pos),
            g[column],
            str(impl),
            styles[str(impl)],
            hollow=hollow,
        )
    marks.append(xy.hline(100, text=ref, color=MUTED, width=1))
    return _line_chart(marks, x_axis, f"% of {ref}", [sub[column], 100], title)


def plot_roofline(sub: pd.DataFrame, styles: dict[str, Style]):
    """Classic roofline: one roof per arithmetic run, impls as points."""
    lo = sub["arithmetic_intensity"].min() * 0.5
    hi = sub["arithmetic_intensity"].max() * 2
    xs = [lo * (hi / lo) ** (i / 200) for i in range(201)]
    bw = sub["mem_gbps"].max()

    marks: list[Any] = []
    for m, peak in sub.groupby("math")["peak_tflops"].first().items():
        ys = [min(peak, bw * x / 1e3) for x in xs]  # GB/s * FLOP/B / 1e3 = TFLOPS
        marks.append(xy.line(xs, ys, name=f"roofline ({m})", color=MUTED, width=2))
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
        *_chrome(),
        _log_axis(
            xy.x_axis,
            [lo, hi],
            "algorithmic arithmetic intensity (FLOP / ideal byte)",
        ),
        _log_axis(xy.y_axis, [*sub["tflops"], *sub["peak_tflops"]], "TFLOPS"),
        xy.legend(loc="lower right"),
        title=f"Roofline — {sub['dtype'].iloc[0]}",
        width=WIDTH,
        height=HEIGHT,
    )


def plot_accuracy_speed(df: pd.DataFrame, metric: Metric, styles: dict[str, Style]):
    """Accuracy cost against speed. Color/symbol follow the impl; the first
    dtype is solid, the others hollow."""
    dtypes = sorted(df["dtype"].unique())
    marks: list[Any] = []
    for (impl, dtype), g in df.groupby(["impl", "dtype"]):
        style = styles[str(impl)]
        hollow = dtype != dtypes[0]
        marks.append(
            xy.scatter(
                list(g[metric.column]),
                list(g["max_abs_diff"].clip(lower=1e-8)),
                name=f"{impl} / {str(dtype).replace('torch.', '')}",
                color=SURFACE if hollow else style.color,
                stroke=style.color if hollow else SURFACE,
                stroke_width=2 if hollow else 1.5,
                symbol=style.symbol,
                size=10,
                opacity=1.0,
            )
        )
    return xy.scatter_chart(
        *marks,
        *_chrome(),
        xy.x_axis(label=metric.unit),
        _log_axis(
            xy.y_axis,
            df["max_abs_diff"].clip(lower=1e-8),
            "max abs diff vs reference",
        ),
        xy.legend(loc="best"),
        title="Accuracy vs throughput",
        width=WIDTH,
        height=HEIGHT,
    )


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


def _op_section(
    op: str, df: pd.DataFrame, baseline: str | None
) -> tuple[list[tuple[str, xy.Chart]], pd.DataFrame, str]:
    """Plots, results table and markdown for one op."""
    failed = df[df["status"] != "ok"]
    # Several sweeps of one op can measure the same case (4096^3 is in both
    # matmul sweeps): keep the newest. Rows arrive in chronological order.
    df = add_derived(df).drop_duplicates(["impl", "dtype", "size_label"], keep="last")
    baseline = resolve_baseline(df, baseline)
    metric = Metric.for_op(df)
    styles = impl_styles(df, baseline)

    base = df[df["impl"] == baseline].set_index(["dtype", "size_label"])[metric.column]
    df["pct_of_baseline"] = (
        100
        * df[metric.column]
        / [base.get((d, s)) for d, s in zip(df["dtype"], df["size_label"])]
    )  # type: ignore

    figs: list[tuple[str, xy.Chart]] = []
    for dtype, sub in df.groupby("dtype"):
        short = str(dtype).replace("torch.", "")
        others = sub[sub["impl"] != baseline]
        figs += [
            (f"{op}_throughput_{short}", plot_throughput(sub, metric, styles)),
            (
                f"{op}_pct_ceiling_{short}",
                plot_pct(
                    sub,
                    "pct_of_ceiling",
                    "ceiling",
                    f"% of ceiling (hollow = memory-bound) — {dtype}",
                    styles,
                    hollow_memory_bound=True,
                ),
            ),
            (
                f"{op}_pct_baseline_{short}",
                plot_pct(
                    others,
                    "pct_of_baseline",
                    baseline,
                    f"% of {baseline} {metric.unit} — {dtype}",
                    styles,
                ),
            ),
            (f"{op}_roofline_{short}", plot_roofline(sub, styles)),
        ]
    figs.append((f"{op}_accuracy_speed", plot_accuracy_speed(df, metric, styles)))

    center, whiskers = quantile_desc(df)
    table = df.sort_values(
        ["dtype", "problem_size", "pct_of_ceiling"], ascending=[True, True, False]
    )[
        [
            "impl",
            "dtype",
            "size_label",
            "t_ms",
            metric.column,
            "pct_of_ceiling",
            "bound",
            "pct_of_baseline",
            "max_abs_diff",
        ]
    ].rename(columns={"t_ms": f"{center}_ms"})

    runs = ", ".join(
        f"{r} ({c})" for r, c in df[["run_id", "cfg_name"]].drop_duplicates().values
    )
    md = (
        f"## {op}\n\n"
        f"runs: {runs}  \n"
        f"git: {', '.join(sorted(df['git_commit'].astype(str).unique()))}  \n"
        f"baseline: {baseline}  \n"
        f"throughput: {metric.unit} "
        f"({(df['bound'] == 'memory').mean():.0%} of cases memory-bound)  \n"
        f"times: {center}" + (f" (spread {whiskers})" if whiskers else "") + "  \n"
        f"cases: {len(df) + len(failed)} ({len(failed)} not ok)\n\n"
        "### Results\n\n"
        + table.to_markdown(index=False, floatfmt=".4g")
        + "\n\n### Plots\n\n"
        + "\n\n".join(f"![{name}](plots/{name}.png)" for name, _ in figs)
    )
    if not failed.empty:
        cols = ["impl", "dtype", "status", "error"]
        md += "\n\n### Not ok\n\n" + failed[cols].to_markdown(index=False)
    return figs, table.assign(op=op), md


def generate_report(
    logs_dir: Path,
    out_dir: Path,
    cfg_name: str | None = None,
    run_id: str | None = None,
    gpu: str | None = None,
    baseline: str | None = None,
) -> Path:
    """One section per op: records of different ops share no params, sizes
    or baseline, so they are never mixed in one table or plot."""
    df = load_records(logs_dir, cfg_name, run_id, gpu)
    out_dir.mkdir(parents=True, exist_ok=True)

    figs: list[tuple[str, xy.Chart]] = []
    tables, sections = [], []
    for op, g in df.groupby("op"):
        op_figs, table, md = _op_section(str(op), g, baseline)
        figs += op_figs
        tables.append(table)
        sections.append(md)

    write_plots(figs, out_dir)
    table = pd.concat(tables, ignore_index=True)
    table[["op", *table.columns.drop("op")]].to_csv(
        out_dir / "summary.csv", index=False
    )

    summary = out_dir / "summary.md"
    summary.write_text(
        f"# {cfg_name or 'all'} — {df['device_name'].iloc[0]}\n\n"
        "plots: [report.pdf](report.pdf)\n\n" + "\n\n".join(sections) + "\n",
        encoding="utf-8",
    )
    return summary
