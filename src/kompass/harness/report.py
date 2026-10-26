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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import xy
from pypdf import PdfWriter
from xy.facets import FacetGrid

# Dense vendor peaks in TFLOPS, keyed by the record's `device_name` (as torch
# and nvidia-smi report it; keys checked with a Modal probe on 2026-10-22) and
# `math` (KernelSpec.math). Sparse datasheet figures are halved.
VENDOR_PEAKS: dict[str, dict[str, float]] = {
    # --- Turing (sm_75) -------------------------------------------------
    # modal gpu="T4". No TF32, no BF16: both postdate Turing. A bf16 benchmark
    # here is emulated, not native -- treat such numbers with suspicion.
    # Source: NVIDIA T4 datasheet (8.1 TFLOPS FP32, 65 TFLOPS FP16 tensor).
    "Tesla T4": {
        "float32": 8.1,
        "float16": 65.0,  # tensor core, dense
    },
    # --- Ampere (sm_80 / sm_86) ----------------------------------------
    # modal gpu="A100-40GB". Modal may serve the 80GB part for this request
    # (the probe got one), so check `device_name` in the records.
    # Source: NVIDIA A100 datasheet; TF32 312 and BF16 624 are quoted WITH
    # sparsity, halved here.
    "NVIDIA A100-SXM4-40GB": {
        "float32": 19.5,
        "tf32": 156.0,
        "bfloat16": 312.0,
        "float16": 312.0,
    },
    # modal gpu="A100-80GB". Same compute as the 40GB part; only HBM differs
    # (1555 -> 2039 GB/s).
    "NVIDIA A100-SXM4-80GB": {
        "float32": 19.5,
        "tf32": 156.0,
        "bfloat16": 312.0,
        "float16": 312.0,
    },
    # modal gpu="A10": the datacenter A10, not the AWS A10G.
    # Source: NVIDIA A10 datasheet; tensor figures quoted with sparsity, halved.
    "NVIDIA A10": {
        "float32": 31.2,
        "tf32": 62.5,
        "bfloat16": 125.0,
        "float16": 125.0,
    },
    # --- Ada Lovelace (sm_89) ------------------------------------------
    # modal gpu="L4".
    # Source: NVIDIA L4 datasheet via Cisco UCS collateral. TF32 120 and BF16
    # 242 are quoted with sparsity ("1/2 lower without sparsity"), halved here.
    "NVIDIA L4": {
        "float32": 30.3,
        "tf32": 60.0,
        "bfloat16": 121.0,
        "float16": 121.0,
    },
    # modal gpu="L40S".
    # Source: NVIDIA L40S datasheet (PNY). Dense values given directly:
    # TF32 183 dense / 366 sparse, BF16 362.05 dense / 733 sparse.
    "NVIDIA L40S": {
        "float32": 91.6,
        "tf32": 183.0,
        "bfloat16": 362.05,
        "float16": 362.05,
    },
    # --- Hopper (sm_90) -------------------------------------------------
    # modal gpu="H100!" (plain "H100" may be upgraded to an H200).
    # Source: NVIDIA H100 datasheet, SXM5. Sparse figures halved.
    "NVIDIA H100 80GB HBM3": {
        "float32": 67.0,
        "tf32": 494.5,
        "bfloat16": 989.4,
        "float16": 989.4,
        "float8": 1978.9,
    },
    # modal gpu="H200". Same GH100 die as H100 SXM, so identical compute;
    # only memory differs (80GB HBM3 @ 3.35 TB/s -> 141GB HBM3e @ 4.8 TB/s).
    "NVIDIA H200": {
        "float32": 67.0,
        "tf32": 494.5,
        "bfloat16": 989.4,
        "float16": 989.4,
        "float8": 1978.9,
    },
    # --- Blackwell (sm_100 / sm_103 / sm_120) ---------------------------
    # modal gpu="B200" / "B300".
    # Sources agree on dense BF16 2250 and TF32 ~1100-1125, but DISAGREE on
    # FP32: flopper.io says 75 dense, verda/datacrunch says 37. That is a
    # factor of 2 (likely dual-issue FP32 counted or not). The conservative
    # value is used; do not quote the fp32 line without checking it against a
    # measurement.
    "NVIDIA B200": {
        "float32": 37.0,  # CONFLICTED: 37 vs 75 across sources
        "tf32": 1100.0,
        "bfloat16": 2250.0,
        "float16": 2250.0,
        "float8": 4500.0,
    },
    "NVIDIA B300 SXM6 AC": {
        "float32": 37.0,  # CONFLICTED, as above
        "tf32": 1100.0,
        "bfloat16": 2250.0,
        "float16": 2250.0,
        "float8": 4500.0,
    },
    # modal gpu="RTX-PRO-6000". The official datasheet quotes only FP32 and
    # peak FP4; it lists no TF32 or BF16 tensor figure, so those keys are
    # deliberately absent rather than guessed.
    # Source: NVIDIA RTX PRO 6000 Blackwell Server Edition datasheet.
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": {
        "float32": 120.0,
    },
}

# GB/s. Kept separate from the TFLOPS table so the units can't be confused --
# the roofline multiplies one by arithmetic intensity and compares to the other.
# Shown next to the measured copy bandwidth, which remains the memory roof.
VENDOR_BANDWIDTH_GBPS: dict[str, float] = {
    "Tesla T4": 320.0,  # GDDR6
    "NVIDIA A100-SXM4-40GB": 1555.0,  # HBM2
    "NVIDIA A100-SXM4-80GB": 2039.0,  # HBM2e
    "NVIDIA A10": 600.0,  # GDDR6
    "NVIDIA L4": 300.0,  # GDDR6
    "NVIDIA L40S": 864.0,  # GDDR6
    "NVIDIA H100 80GB HBM3": 3350.0,  # HBM3
    "NVIDIA H200": 4800.0,  # HBM3e
    "NVIDIA B200": 7700.0,  # HBM3e; some sources say 8000
    "NVIDIA B300 SXM6 AC": 8000.0,  # HBM3e; conflicts with 7700 elsewhere
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 1597.0,  # GDDR7
}

SURFACE = "#FEFEFE"
GRID = "#EAEAEA"
TEXT = "#000000"
MUTED = "#020202A2"  # reference lines / roofline / baseline: never a palette hue
PALETTE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


SYMBOLS = ["circle", "square", "diamond", "triangle", "hexagon", "pentagon", "star"]

WIDTH, HEIGHT = 1000, 520

TUNED = " (tuned)"  # suffix of an impl run with its `kompass tune` config


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
    df = df[df["kind"] == "bench"]  # `tune` trials share the logs and cfg_name
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


def _peak(gpu: str, math: str) -> float:
    if math not in VENDOR_PEAKS.get(gpu, {}):
        raise ValueError(
            f"no vendor peak for {math!r} on {gpu!r}: add it to VENDOR_PEAKS in "
            "harness/report.py"
        )
    return VENDOR_PEAKS[gpu][math]


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
    df["peak_tflops"] = [_peak(gpu, m) for gpu, m in zip(df["device_name"], df["math"])]
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
    dashed: bool = False

    @property
    def size(self) -> float:
        return 12 if self.symbol == "triangle" else 9  # triangles render smaller


def impl_styles(df: pd.DataFrame, baseline: str) -> dict[str, Style]:
    """One color/symbol per impl; a tuned twin shares its impl's, drawn dashed."""
    impls = df["impl"].unique()
    bases = sorted({i.removesuffix(TUNED) for i in impls} - {baseline})
    base_styles = {
        b: Style(PALETTE[k % len(PALETTE)], SYMBOLS[k % len(SYMBOLS)])
        for k, b in enumerate(bases)
    }
    base_styles[baseline] = Style(MUTED, "circle")
    return {
        i: replace(base_styles[i.removesuffix(TUNED)], dashed=i.endswith(TUNED))
        for i in impls
    }


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


def _varying_params(sub: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(varying, fixed) `param_*` columns of `sub`; all of them vary if none do."""
    params = [c for c in sub.columns if c.startswith("param_") and sub[c].notna().any()]
    varying = [c for c in params if sub[c].nunique() > 1] or params
    return varying, [c for c in params if c not in varying]


def _size_axis(sub: pd.DataFrame) -> tuple[dict[str, int], Any]:
    """Ordinal x positions for problem sizes, ordered by size (not by string),
    plus the matching x axis. Ordinal rather than categorical so every series
    shares one order even when an impl is missing a size.

    Ticks show only the params that vary (e.g. `N` in a K-sweep); the fixed
    ones go in the axis title.
    """
    varying, fixed = _varying_params(sub)

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


def _log_axis(
    axis: Any, values: Any, label: str, extra: dict[float, str] | None = None
) -> Any:
    """Log axis with 1-2-5 ticks over the data range (xy labels only decades
    by default, which leaves a 1.5-decade roofline with one tick). `extra`
    adds labelled ticks, e.g. {1e-9: "exact"}."""
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
    labels = {t: f"{t:g}" for t in ticks} | (extra or {})
    return axis(
        label=label,
        type_="log",
        domain=(lo, hi),
        tick_values=list(labels),
        tick_labels=list(labels.values()),
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
            size=style.size,
            opacity=1.0,
            name=None if lines else name,
        )
        for fill in (False, True)
        if fill in hollow
    ]
    if lines:
        dash = (6, 4) if style.dashed else None
        marks.insert(
            0, xy.line(x, y, name=name, color=style.color, width=2.5, dash=dash)
        )
    return marks


def _line_chart(marks: list[Any], x_axis: Any, y_axis: Any, title: str):
    return xy.line_chart(
        *marks,
        *_chrome(),
        x_axis,
        y_axis,
        xy.legend(loc="best"),
        title=title,
        width=WIDTH,
        height=HEIGHT,
    )


def plot_throughput(
    sub: pd.DataFrame,
    scope: pd.DataFrame,
    metric: Metric,
    styles: dict[str, Style],
    title: str,
):
    """Throughput vs problem size, with the ceilings of this unit. Log y:
    throughput spans decades across sizes, and peaks sit far above the data.
    Axes span `scope` so facet panels line up."""
    pos, x_axis = _size_axis(scope)
    center, whiskers = quantile_desc(scope)
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
                    # clip: tied quantiles round to ~-1e-16 instead of 0
                    yerr=(
                        list((g[col] - lo).clip(lower=0)),
                        list((hi - g[col]).clip(lower=0)),
                    ),
                    color=styles[str(impl)].color,
                    cap_size=0.08,  # data units (ordinal positions), not px
                    opacity=0.7,
                )
            )
        marks += _series(x, g[col], str(impl), styles[str(impl)])

    def roofs(rows: pd.DataFrame) -> dict[str, Any]:
        if metric.column == "gbps":
            gpu = rows["device_name"].iloc[0]
            bw = {"measured copy bandwidth": rows["mem_gbps"].max()}
            if gpu in VENDOR_BANDWIDTH_GBPS:
                bw["vendor bandwidth"] = VENDOR_BANDWIDTH_GBPS[gpu]
            return bw
        peaks = rows.groupby("math")["peak_tflops"].first()
        return {f"{m} peak": v for m, v in peaks.items()}

    for text, v in roofs(sub).items():
        marks.append(xy.hline(v, text=f"{text} ({v:.0f})", color=MUTED, width=1))

    y_axis = _log_axis(
        xy.y_axis,
        [*(scope[col] * scope["t_ms"] / scope["t_hi_ms"]), *roofs(scope).values()],
        f"{metric.unit} ({center}" + (f", whiskers {whiskers})" if whiskers else ")"),
    )
    return _line_chart(marks, x_axis, y_axis, title)


def plot_pct(
    sub: pd.DataFrame,
    scope: pd.DataFrame,
    column: str,
    ref: str,
    title: str,
    styles: dict[str, Style],
    hollow_memory_bound: bool = False,
    line_text: str | None = None,
):
    """Each impl as a percentage of a reference (the baseline, or its own
    ceiling), per size, with the 100% line labelled `line_text` (default
    `ref`). Axes span `scope`."""
    pos, x_axis = _size_axis(scope)
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
    marks.append(xy.hline(100, text=line_text or ref, color=MUTED, width=1))
    y_axis = xy.y_axis(label=f"% of {ref}", domain=_zero_based(scope[column], 100))
    return _line_chart(marks, x_axis, y_axis, title)


def plot_roofline(
    sub: pd.DataFrame, scope: pd.DataFrame, styles: dict[str, Style], title: str
):
    """Classic roofline: one roof per arithmetic run, impls as points. Axes
    span `scope`."""
    lo = scope["arithmetic_intensity"].min() * 0.5
    hi = scope["arithmetic_intensity"].max() * 2
    xs = [lo * (hi / lo) ** (i / 200) for i in range(201)]
    bw = scope["mem_gbps"].max()

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
    # Name sizes (varying params only) above their highest point, skipping any
    # closer than 4% of the axis to the last label: intensity flattens at large
    # sizes, and sizes can share one intensity entirely (e.g. softmax).
    varying, _ = _varying_params(scope)
    tops = sub.groupby("size_label").agg(
        x=("arithmetic_intensity", "first"),
        y=("tflops", "max"),
        **{c: (c, "first") for c in varying},
    )
    last = -math.inf
    for r in sorted(tops.to_dict("records"), key=lambda r: r["x"]):  # type: ignore
        if math.log10(r["x"]) - last < 0.04 * math.log10(hi / lo):
            continue
        last = math.log10(r["x"])
        text = "x".join(str(int(r[c])) for c in varying)
        marks.append(
            xy.text(r["x"], r["y"], text, color=MUTED, anchor="middle", dx=0, dy=-12)
        )

    return xy.scatter_chart(
        *marks,
        *_chrome(),
        _log_axis(
            xy.x_axis,
            [lo, hi],
            "algorithmic arithmetic intensity (FLOP / ideal byte)",
        ),
        _log_axis(xy.y_axis, [*scope["tflops"], *scope["peak_tflops"]], "TFLOPS"),
        xy.legend(loc="lower right"),
        title=title,
        width=WIDTH,
        height=HEIGHT,
    )


def plot_accuracy_speed(
    sub: pd.DataFrame,
    scope: pd.DataFrame,
    metric: Metric,
    styles: dict[str, Style],
    title: str,
):
    """Accuracy cost against speed, one point per impl (callers pass one size,
    so the x position compares impls, not problem sizes). Exact results
    (max abs diff 0) sit on an `exact` row a decade below the smallest error
    in `scope`, rather than at a made-up value."""
    errors = scope["max_abs_diff"]
    exact = errors[errors > 0].min() / 10 if (errors > 0).any() else 1e-8
    marks: list[Any] = []
    for impl, g in sub.groupby("impl"):
        style = styles[str(impl)]
        marks.append(
            xy.scatter(
                list(g[metric.column]),
                list(g["max_abs_diff"].where(g["max_abs_diff"] > 0, exact)),
                name=str(impl),
                color=style.color,
                stroke=SURFACE,
                stroke_width=1.5,
                symbol=style.symbol,
                size=style.size + 3,
                opacity=1.0,
            )
        )
    return xy.scatter_chart(
        *marks,
        *_chrome(),
        xy.x_axis(label=metric.unit, domain=_zero_based(scope[metric.column])),
        _log_axis(
            xy.y_axis,
            [*errors[errors > 0], exact],
            "max abs diff vs reference",
            extra={exact: "exact"},
        ),
        xy.legend(loc="best"),
        title=title,
        width=WIDTH,
        height=HEIGHT,
    )


def _ceiling_label(rows: pd.DataFrame) -> str:
    """What 100% of ceiling means for `rows` (one `math`): its compute peak,
    the copy bandwidth, or the min of both when bounds are mixed."""
    m, peak, bw = (
        rows["math"].iloc[0],
        rows["peak_tflops"].max(),
        rows["mem_gbps"].max(),
    )
    bounds = set(rows["bound"])
    if bounds == {"compute"}:
        return f"ceiling: {m} peak ({peak:.0f} TFLOPS)"
    if bounds == {"memory"}:
        return f"ceiling: copy bandwidth ({bw:.0f} GB/s)"
    return f"ceiling: min({m} peak {peak:.0f} TFLOPS, {bw:.0f} GB/s x intensity)"


def _by_math(rows: pd.DataFrame, title: str, plot: Any) -> Any:
    """`plot(rows, scope, title)` once per `math`, side by side on one page:
    impls of one dtype can run different arithmetic (float32 vs tf32), each
    under its own ceiling. Panels share `rows` as scope, so axes line up."""
    maths = sorted(rows["math"].unique())
    if len(maths) == 1:
        return plot(rows, rows, title)
    panels = [plot(rows[rows["math"] == m], rows, f"{title}, {m} math") for m in maths]
    return FacetGrid(
        [p.figure() for p in panels],
        maths,
        cols=len(maths),
        width=WIDTH * len(maths),
        height=HEIGHT,
    )


def write_plots(figs: list[tuple[str, Any]], out_dir: Path) -> Path:
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


def _sweep_section(
    op: str, cfg: str, df: pd.DataFrame, baseline: str | None
) -> tuple[list[tuple[str, Any]], pd.DataFrame, str]:
    """Plots, results table and markdown for one sweep (one op, one config)."""
    if "tuning" in df:  # tuned twins (`bench --tuning both`) are their own series
        tuned = df["tuning"].map(lambda t: isinstance(t, dict) and bool(t))
        df = df.assign(impl=df["impl"].where(~tuned, df["impl"] + TUNED))
    failed = df[df["status"] != "ok"]
    df = add_derived(df)
    baseline = resolve_baseline(df, baseline)
    metric = Metric.for_op(df)
    styles = impl_styles(df, baseline)

    # Compare like with like: the baseline running the same arithmetic.
    keys = ["dtype", "math", "size_label"]
    base = df[df["impl"] == baseline].set_index(keys)[metric.column]
    df["pct_of_baseline"] = (
        100 * df[metric.column] / [base.get(k) for k in zip(*(df[c] for c in keys))]
    )  # type: ignore
    is_tuned = df["impl"].str.endswith(TUNED)
    if is_tuned.any():  # what tuning gained, against the same impl untuned
        own = df.set_index(["impl", *keys])[metric.column]
        twins = zip(df["impl"].str.removesuffix(TUNED), *(df[c] for c in keys))
        df["pct_of_untuned"] = (
            100 * df[metric.column] / [own.get(k) for k in twins]
        ).where(is_tuned)  # type: ignore

    figs: list[tuple[str, Any]] = []
    for dtype, sub in df.groupby("dtype"):
        short = str(dtype).replace("torch.", "")
        head = f"{op} / {cfg} —"
        # rows whose arithmetic the baseline didn't run have no % of baseline
        vs_base = sub[(sub["impl"] != baseline) & sub["pct_of_baseline"].notna()]
        figs += [
            (
                f"{cfg}_throughput_{short}",
                _by_math(
                    sub,
                    f"{head} Throughput — {short}",
                    lambda rows, scope, title: plot_throughput(
                        rows, scope, metric, styles, title
                    ),
                ),
            ),
            (
                f"{cfg}_pct_ceiling_{short}",
                _by_math(
                    sub,
                    f"{head} % of ceiling (hollow = memory-bound) — {short}",
                    lambda rows, scope, title: plot_pct(
                        rows,
                        scope,
                        "pct_of_ceiling",
                        "ceiling",
                        title,
                        styles,
                        hollow_memory_bound=True,
                        line_text=_ceiling_label(rows),
                    ),
                ),
            ),
            (
                f"{cfg}_roofline_{short}",
                _by_math(
                    sub,
                    f"{head} Roofline — {short}",
                    lambda rows, scope, title: plot_roofline(
                        rows, scope, styles, title
                    ),
                ),
            ),
        ]
        if not vs_base.empty:
            figs.append(
                (
                    f"{cfg}_pct_baseline_{short}",
                    _by_math(
                        vs_base,  # type: ignore
                        f"{head} % of {baseline} {metric.unit} — {short}",
                        lambda rows, scope, title: plot_pct(
                            rows, scope, "pct_of_baseline", baseline, title, styles
                        ),
                    ),
                )
            )
    # The largest size per arithmetic, where speed and error have settled.
    largest = df[
        df["problem_size"] == df.groupby("math")["problem_size"].transform("max")
    ]
    sizes = ", ".join(sorted(largest["size_label"].unique()))  # type: ignore
    figs.append(
        (
            f"{cfg}_accuracy_speed",
            _by_math(
                largest,  # type: ignore
                f"{op} / {cfg} — Accuracy vs throughput at {sizes}",
                lambda rows, scope, title: plot_accuracy_speed(
                    rows, scope, metric, styles, title
                ),
            ),
        )
    )

    center, whiskers = quantile_desc(df)
    table = df.sort_values(
        ["dtype", "math", "problem_size", "pct_of_ceiling"],
        ascending=[True, True, True, False],
    )[
        [
            "impl",
            "dtype",
            "math",
            "size_label",
            "t_ms",
            metric.column,
            "pct_of_ceiling",
            "bound",
            "pct_of_baseline",
            *(["pct_of_untuned"] if "pct_of_untuned" in df else []),
            "max_abs_diff",
        ]
    ].rename(columns={"t_ms": f"{center}_ms"})
    table["dtype"] = table["dtype"].str.removeprefix("torch.")

    runs = ", ".join(
        f"{r} ({c})" for r, c in df[["run_id", "cfg_name"]].drop_duplicates().values
    )
    md = (
        f"## {op} / {cfg}\n\n"
        f"runs: {runs}  \n"
        f"git: {', '.join(sorted(df['git_commit'].astype(str).unique()))}  \n"
        f"baseline: {baseline}  \n"
        f"throughput: {metric.unit} "
        f"({(df['bound'] == 'memory').mean():.0%} of cases memory-bound)  \n"
        f"times: {center}" + (f" (spread {whiskers})" if whiskers else "") + "  \n"
        f"cases: {len(df) + len(failed)} ({len(failed)} not ok)\n\n"
        "### Results\n\n"
        + table.astype(object)
        .where(table.notna(), None)
        .to_markdown(index=False, floatfmt=".4g", missingval="")
        + "\n\n### Plots\n\n"
        + "\n\n".join(f"![{name}](plots/{name}.png)" for name, _ in figs)
    )
    if not failed.empty:
        cols = ["impl", "dtype", "status", "error"]
        md += "\n\n### Not ok\n\n" + failed[cols].to_markdown(index=False)
    return figs, table.assign(op=op, cfg_name=cfg), md


def generate_report(
    logs_dir: Path,
    out_dir: Path,
    cfg_name: str | None = None,
    run_id: str | None = None,
    gpu: str | None = None,
    baseline: str | None = None,
) -> Path:
    """One section per sweep (op, cfg_name): different sweeps share no x axis,
    even for the same op, so they are never mixed in one table or plot."""
    df = load_records(logs_dir, cfg_name, run_id, gpu)
    out_dir.mkdir(parents=True, exist_ok=True)

    figs: list[tuple[str, Any]] = []
    tables, sections = [], []
    for (op, cfg), g in df.groupby(["op", "cfg_name"]):  # type: ignore
        op_figs, table, md = _sweep_section(str(op), str(cfg), g, baseline)
        figs += op_figs
        tables.append(table)
        sections.append(md)

    write_plots(figs, out_dir)
    table = pd.concat(tables, ignore_index=True)
    table[["op", "cfg_name", *table.columns.drop(["op", "cfg_name"])]].to_csv(
        out_dir / "summary.csv", index=False
    )

    summary = out_dir / "summary.md"
    summary.write_text(
        f"# {cfg_name or 'all'} — {df['device_name'].iloc[0]}\n\n"
        "plots: [report.pdf](report.pdf)\n\n" + "\n\n".join(sections) + "\n",
        encoding="utf-8",
    )
    return summary
