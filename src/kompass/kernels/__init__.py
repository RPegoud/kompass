from ..harness.registry import OpEntry
from .softmax import REGISTRY as _softmax
from .vec_add import REGISTRY as _vec_add

REGISTRY: dict[str, OpEntry] = {}

for r in [_vec_add, _softmax]:
    REGISTRY.update(r)
