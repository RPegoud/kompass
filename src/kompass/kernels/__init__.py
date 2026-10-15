from ..harness.registry import OpEntry
from .matmul import REGISTRY as _matmul
from .softmax import REGISTRY as _softmax
from .vec_add import REGISTRY as _vec_add

REGISTRY: dict[str, OpEntry] = {}

for r in [_vec_add, _softmax, _matmul]:
    REGISTRY.update(r)
