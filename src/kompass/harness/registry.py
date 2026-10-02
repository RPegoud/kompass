from dataclasses import dataclass, field

from .ops import Op
from .spec import KernelSpec


@dataclass(frozen=True)
class OpEntry:
    op: Op
    impls: dict[str, KernelSpec] = field(default_factory=dict)
