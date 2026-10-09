from dataclasses import dataclass


@dataclass
class ProfileRecord:
    meta: dict[str, str | int]
    kernel_name: str
    action_idx: int
    duration_ns: float
    dram_bytes: int
    dram_pct_peak: float
    sm_pct_peak: float
