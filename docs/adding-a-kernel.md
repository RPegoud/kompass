# 🧩 Adding a kernel

[← Back to the README](../README.md)

Every Op lives in its own package under `src/kompass/kernels/<op>/`. The walkthrough below
uses `vec_add`. The same steps apply to any new Op.

```
src/kompass/kernels/vec_add/
├── __init__.py   # Op definition + KernelSpecs + REGISTRY
├── vec_add.py    # Triton kernel + Python launcher
└── vecAdd.cu     # CUDA kernel + pybind11 binding
```

## 📐 1. Define the Op

The Op is the contract that every implementation is tested and measured against:

```python
# kernels/vec_add/__init__.py
class VecAdd(Op):
    param_names: tuple[str, ...] = ("N",)   # keys used in the sweep YAML's `axes` / `shapes`

    @staticmethod
    def reference(*tensors, params=None) -> torch.Tensor:
        A, B = tensors
        return torch.add(A, B)

    def compare(self, pred, ref) -> TestResult:
        return TestResult(
            correct=torch.equal(pred, ref),          # use torch.testing.assert_close for reductions
            max_abs_diff=torch.max(torch.abs(ref - pred)).item(),
        )

    @staticmethod
    def make_inputs(params, dtype, requires_grad, seed, device):
        gen = torch.Generator(device).manual_seed(seed)
        A = torch.randn(params["N"], generator=gen, dtype=dtype, device=device, requires_grad=requires_grad)
        B = torch.randn(params["N"], generator=gen, dtype=dtype, device=device, requires_grad=requires_grad)
        return A, B

    @staticmethod
    def bytes_moved(params, dtype) -> int:   # minimum DRAM traffic: read A, B, write C
        return 3 * params["N"] * dtype.itemsize

    @staticmethod
    def flops(params) -> int:
        return params["N"]
```

`bytes_moved` and `flops` give the arithmetic intensity used in the roofline report.
`bytes_moved` is also used to skip cases that wouldn't fit in GPU memory.

## 🔱 2a. Write a Triton kernel

A Triton impl is a plain Python function that takes the tensors returned by `make_inputs`
and returns the output:

```python
# kernels/vec_add/vec_add.py
@triton.jit
def vec_add_fwd_kernel(X_ptr, Y_ptr, Z_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    X = tl.load(X_ptr + offs, mask=mask, other=0.0)
    Y = tl.load(Y_ptr + offs, mask=mask, other=0.0)
    tl.store(Z_ptr + offs, X + Y, mask=mask)


def vec_add_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    C = torch.empty_like(A)
    n = C.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    vec_add_fwd_kernel[grid](A, B, C, n, BLOCK_SIZE=256)
    return C
```

## ⚡ 2b. Write a CUDA kernel

A CUDA impl is a `.cu` file that exposes a function through pybind11. `CudaLoader` compiles it
with `torch.utils.cpp_extension.load`, caches the build per GPU arch (`sm_XY`) and per source hash,
and adds the `kernels/` directory to the include path so `../common.cuh` resolves.

```cpp
// kernels/vec_add/vecAdd.cu
#include "../common.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

__global__ void vecAddFwdKernel(float *A, float *B, float *C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) C[i] = A[i] + B[i];
}

at::Tensor vec_add_fwd(const at::Tensor &A, const at::Tensor &B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kFloat, "float32 only");
  const at::cuda::CUDAGuard guard(A.device());

  auto C = at::empty_like(A);
  const int n = A.numel(), threads = 256;
  vecAddFwdKernel<<<(n + threads - 1) / threads, threads>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n);
  CUDA_CHECK(cudaGetLastError());
  return C;
}

// The name bound here must equal CudaLoader's `fn_name`
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("vec_add_cuda", &vec_add_fwd); }
```

## 📝 3. Register the implementations

Wrap each implementation in a `KernelSpec` (Op + Loader) and expose a `REGISTRY`:

```python
# kernels/vec_add/__init__.py
VecAddSpec = partial(KernelSpec, op=VecAdd())

vec_add_torch = VecAddSpec(loader=TorchLoader(fn=torch.add))

vec_add_triton = VecAddSpec(
    loader=TritonLoader(
        module_path=f"{__package__}.vec_add",  # module that holds the launcher
        fn_name="vec_add_triton",              # launcher function name
    ),
)

vec_add_cuda = VecAddSpec(
    loader=CudaLoader(
        module_name="vec_add_cuda",            # extension name (arch + hash are appended)
        fn_name="vec_add_cuda",                # name bound in PYBIND11_MODULE
        sources=[str(Path(__file__).parent / "vecAdd.cu")],
    ),
)
vec_add_torch = VecAddSpec(                    # torch reference
    loader=TorchLoader(fn=torch.add),          # simply pass the raw function
)
REGISTRY: dict[str, OpEntry] = {
    "vec_add": OpEntry(
        op=VecAdd(),
        impls={
            "vec_add_torch": vec_add_torch,
            "vec_add_triton": vec_add_triton,
            "vec_add_cuda": vec_add_cuda,
        },
    )
}
```

Name impls `<op>_<suffix>` so that `kompass profile <op>/<suffix>` can find them.

If a launcher takes keyword arguments that a sweep should vary, register one impl and list them in
`option_names` instead of registering one impl per value. Each name must also be in the Op's
`option_names`:

```python
# kernels/matmul/__init__.py
matmul_triton_tiled = MatmulSpec(
    loader=TritonLoader(module_path=f"{__package__}.matmul", fn_name="matmul_triton_tiled"),
    option_names=("input_precision",),         # matmul_triton_tiled(X, Y, input_precision="ieee")
)
```
For a new Op, also merge its registry into the global one in
[kernels/\_\_init\_\_.py](../src/kompass/kernels/__init__.py):

```python
from .vec_add import REGISTRY as _vec_add
from .my_op import REGISTRY as _my_op

for r in [_vec_add, ..., _my_op]:
    REGISTRY.update(r)
```

## 🧪 4. Describe a sweep

A sweep YAML names the Op, the impls and dtypes to compare, and the parameter sets to run.
Kompass runs every `impl × dtype × parameter set` combination. You define the parameter sets
in one of two ways, and a config must use exactly one of them:

- **`axes`** gives one list of values per parameter. Kompass runs the **Cartesian product** of the lists.
  Use it for scans, e.g. sweeping `N` for `vec_add`, or sweeping `K` with `M` and `N` fixed for `matmul`.
- **`shapes`** gives an **explicit list** of parameter combinations. Only those are run. Use it when
  parameters move together (e.g. square matmuls), so the full grid would be mostly irrelevant cases.

Every parameter set must define exactly the Op's `param_names`. Otherwise `bench` exits with an error.

**`options`** (optional) sweeps kernel keyword arguments, e.g. `input_precision: [ieee, tf32]` for matmul.
Keys must be in the Op's `option_names`. Each impl runs every combination of the options it declares
in its `KernelSpec(option_names=...)`; impls that declare none run once. `matmul_torch` declares
`input_precision` too, so each precision is compared against torch running the same arithmetic.
Combinations that don't change the arithmetic on a dtype run once: `tf32` only affects float32
inputs, so bf16 runs each matmul kernel once. The report puts each arithmetic (`float32`, `tf32`, ...)
in its own panel, side by side on the same page.

```yaml
# src/kompass/cases/softmax_occupancy.yaml (grid via `axes`)
op: softmax
impls: [softmax_triton_online, softmax_torch]
axes:  # one list per entry in Op.param_names
  M: [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384] 
  N: [4096]
dtypes: [float32]
seed: 0
quantiles: [0.2, 0.5, 0.8] # passed to do_bench
```

```yaml
# src/kompass/cases/matmul_square.yaml (explicit combinations via `shapes`)
op: matmul
impls: [matmul_triton_tiled, matmul_torch]
options:
  input_precision: [ieee, tf32] # float32 only: bf16 runs once per impl
shapes:
  - { M: 1024, N: 1024, K: 1024 }
  - { M: 4096, N: 4096, K: 4096 }
  - { M: 8192, N: 8192, K: 8192 }
dtypes: [float32, bfloat16]
seed: 0
quantiles: [0.2, 0.5, 0.8]
```

With `axes: {M: [1024, 4096], N: [1024, 4096], K: [1024, 4096]}` you would get 8 shapes.
The `shapes` list above runs only the 3 square ones.

## 🚀 5. Run it

```bash
kompass list                                              # the new impls should be listed
kompass bench vec_add_sweep                               # correctness-gated benchmark
kompass profile vec_add/cuda -p N=16777216                # Nsight Compute deep dive
```

## 🔧 6. Make it tunable

To let `tune` search a kernel's launch parameters:

1. Have the launcher take them as keyword arguments, with the defaults `bench` uses when no tuned
   config exists.
2. Give its `KernelSpec` a `search_space` (one list of values per keyword) and, optionally, a `prune`
   hook that rejects configs before they're compiled.

```python
# kernels/matmul/matmul.py
def matmul_triton_tiled(X, Y, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                        num_stages=3, num_warps=8, input_precision="ieee"): ...

# kernels/matmul/__init__.py
def prune_matmul_config(params, tuning, dtype, limits) -> tuple[bool, str | None]:
    """(skip, reason). `limits` is the worker's describe_env(), e.g. its shared memory."""
    t = tuning
    smem = t["num_stages"] * (t["BLOCK_SIZE_M"] + t["BLOCK_SIZE_N"]) * t["BLOCK_SIZE_K"] * dtype.itemsize
    if smem > 0.9 * limits["shared_memory_per_block_optin"]:
        return True, f"needs ~{smem // 1024}KB of shared memory"
    return False, None

matmul_triton_tiled = MatmulSpec(
    loader=TritonLoader(module_path=f"{__package__}.matmul", fn_name="matmul_triton_tiled"),
    option_names=("input_precision",),
    search_space={
        "BLOCK_SIZE_M": [64, 128, 256], "BLOCK_SIZE_N": [64, 128, 256], "BLOCK_SIZE_K": [32, 64, 128],
        "num_warps": [4, 8], "num_stages": [2, 3, 4],
    },
    prune=prune_matmul_config,
)
```

`tune` runs the sweep's cases once per config in the grid:

- **Pass 1** times every config briefly. A config whose first call is already more than 3× slower than
  the best one found so far (and over 1 ms) is recorded as `slow` and not timed in full.
- **Pass 2** re-times the 5 fastest configs per case, on the same GPU.
- **Failures don't stop the run:** configs that exceed the GPU's shared memory get status `oom_smem`,
  and wrong results get `incorrect`; both are recorded like any other case.

Tuned configs are specific to a GPU and a shape, and `bench` only uses a config saved for the exact
same dtype, options and shape. Commit the files in `src/kompass/tuning/` to share them: the Modal image
ships them with the code.

## 📂 Outputs

```
logs/<timestamp>__<op>__<run_id>/   # bench
├── meta.json        # run config, GPU/driver/nvcc info, measured copy bandwidth, git commit (or `error` if the sweep failed)
└── times.jsonl      # one row per case (times at each quantile, flops, bytes, status, memory, tuning used)

logs/<timestamp>__<op>__<run_id>/   # tune (kind: "tune", ignored by `report`)
├── meta.json        # pruned configs with their reason, failed groups
└── times.jsonl      # one row per trial, with its `tuning` and `stage` (1: short pass, 2: re-timed finalists)

src/kompass/tuning/<gpu>/<impl>.json   # tune: best config per (dtype, options, shape), read by bench

logs/<timestamp>__<op>__<uid>/      # profile (ingest)
├── meta.json
└── records.jsonl    # one row per kernel launch (duration, DRAM bytes, %DRAM / %SM of peak)

reports/<op>_<impl>/<uid>/          # profile
├── meta.json
└── report.ncu-rep   # open in Nsight Compute UI

reports/<config_name>/              # report
├── summary.md       # per-sweep results table + embedded plots
├── summary.csv
├── plots/*.png
└── report.pdf       # every plot, one per page
```

Every row is flat: params appear as `param_<name>` columns. That makes loading a run a
one-liner, e.g. `pd.read_json("logs/.../times.jsonl", lines=True)`.

## 💬 Tips

- **Correctness first.** A failing `compare` marks the case `incorrect` and skips timing.
  Check the `error` field in `times.jsonl` for the max abs diff.
- **Compilation cache.** CUDA builds are keyed on `sm_XY` plus a hash of the sources, so editing a
  `.cu` file triggers a rebuild. The devcontainer keeps `ccache` and `torch_extensions` on volumes,
  and Modal keeps them on the `cuda_kernels` and `triton_kernels` volumes.
- **Autograd.** Set `supports_backward=True` on a loader to have `make_inputs` create tensors
  with `requires_grad=True`.
