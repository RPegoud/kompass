from ..harness.registry import OpEntry
from .matmul import REGISTRY as _matmul
from .softmax import REGISTRY as _softmax
from .vec_add import REGISTRY as _vec_add

REGISTRY: dict[str, OpEntry] = {}

for r in [_vec_add, _softmax, _matmul]:
    REGISTRY.update(r)


def _check_names() -> None:
    """Impl keys follow `{op}_{framework}[_{details}]`, e.g.
    `matmul_triton_tiled_l2_swizzle`. Records and reports rely on it."""
    for op_name, entry in REGISTRY.items():
        for key, spec in entry.impls.items():
            prefix = f"{op_name}_{spec.loader.framework}"
            if key != prefix and not key.startswith(prefix + "_"):
                raise ValueError(
                    f"impl {key!r} should be named {prefix}[_details] "
                    f"(op {op_name!r}, framework {spec.loader.framework!r})"
                )


_check_names()
