from ..harness.registry import OpEntry
from .vec_add import REGISTRY as _vec_add

REGISTRY: dict[str, OpEntry] = {}

for r in [_vec_add]:
    REGISTRY.update(r)
