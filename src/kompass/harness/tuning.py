"""Best tuning configs found by `kompass tune`, one JSON file per GPU and impl
under `src/kompass/tuning/`, applied by `kompass bench`. They live inside the
package so the Modal image ships them; commit them like the sweep configs.

A config is tuned for one (dtype, options, params) case: lookups are exact, and
a shape that wasn't tuned runs with the launcher's defaults.
"""

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from .schema import Case

TUNING_DIR = Path(__file__).resolve().parents[1] / "tuning"


def _path(device_name: str, impl: str) -> Path:
    return TUNING_DIR / device_name.replace(" ", "_") / f"{impl}.json"


def _key(dtype: str, options: dict[str, str], params: dict[str, int]) -> str:
    """The case a config is tuned for. Result rows carry `torch.float32`, cases
    `float32`: both map to the same key."""
    case = {"dtype": dtype.removeprefix("torch."), "options": options, "params": params}
    return json.dumps(case, sort_keys=True)


def median_ms(times: list[float], quantiles: list[float]) -> float:
    """Time at the quantile closest to 0.5 (`times` follows `quantiles`)."""
    i = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))
    return times[i]


def load(device_name: str, impl: str) -> dict[str, dict[str, Any]]:
    path = _path(device_name, impl)
    if not path.exists():
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {_key(e["dtype"], e["options"], e["params"]): e for e in entries}


def save(
    rows: list[dict[str, Any]], device_name: str, run_id: str, git_commit: str | None
) -> list[Path]:
    """Merge the fastest stage-2 config per (impl, dtype, options, params) of
    `rows` (un-flattened `CaseResult` dicts from `run_tune`) into the files of
    `device_name`, replacing older entries for the same case."""
    best: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r["stage"] != 2 or r["status"] != "ok":
            continue
        entry = {
            "dtype": r["dtype"].removeprefix("torch."),
            "options": r["options"],
            "params": r["params"],
            "tuning": r["tuning"],
            "median_ms": median_ms(r["times"], r["quantiles"]),
            "run_id": run_id,
            "git_commit": git_commit,
        }
        key = _key(r["dtype"], r["options"], r["params"])
        current = best[r["impl"]].get(key)
        if current is None or entry["median_ms"] < current["median_ms"]:
            best[r["impl"]][key] = entry

    paths = []
    for impl, entries in best.items():
        merged = load(device_name, impl) | entries
        path = _path(device_name, impl)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([merged[k] for k in sorted(merged)], indent=2) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def apply(cases: list[Case], device_name: str) -> list[Case]:
    """`cases` with the saved config of `device_name` filled into those without
    `tuning`; cases with no saved config are returned unchanged."""
    saved: dict[str, dict[str, dict[str, Any]]] = {}
    out = []
    for case in cases:
        if case.impl not in saved:
            saved[case.impl] = load(device_name, case.impl)
        entry = saved[case.impl].get(_key(case.dtype, case.options, case.params))
        if entry and not case.tuning:
            case = replace(case, tuning=entry["tuning"])
        out.append(case)
    return out
