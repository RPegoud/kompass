# `Kompass`

<a href="Kompass.png?raw=true"><img src="Kompass.png?raw=true" width="200" align="right" /></a>

**Kompass is a tooling suite for GPU kernel development.** It gives you one harness to
**test, benchmark and profile** CUDA, Triton and PyTorch implementations of the same
operation side by side. Runs can happen on your local GPU or on a cloud GPU through
[Modal](https://modal.com).

You describe an **Op** once (how to build inputs, a reference result, a correctness check,
and FLOP/byte counts). You then register any number of **implementations** of it. Kompass:

- **JIT-compiles** CUDA extensions per GPU architecture and source hash, and imports Triton kernels lazily.
- **Gates every benchmark on correctness**: a kernel whose output doesn't match the reference is never timed.
- **Sweeps** parameter grids from a YAML file and records each case's status (`ok`, `incorrect`, `oom`, `skipped_oom`, `failed`).
- **Measures copy bandwidth** on the target GPU, so results can be compared against an achievable roofline.
- **Profiles** a single case under Nsight Compute and turns the `.ncu-rep` into flat JSONL records.
- Stamps every run with the git commit (`-dirty` if there are uncommitted changes) and the GPU, driver and CUDA versions, so results stay reproducible.

---

## CLI endpoints

After `uv sync`, all endpoints are available through the `kompass` entry point.

| Command | What it does |
| --- | --- |
| `kompass list` | Prints every registered Op and its implementations. |
| `kompass bench <cases/*.yaml> [--backend local\|modal:<gpu>] [--device cuda:0]` | Runs a sweep: every `impl × dtype × param` combination in the YAML is checked for correctness, then timed with `triton.testing.do_bench`. |
| `kompass profile <op>/<impl> -p K=V [-p K=V ...] [--dtype float32] [--set full\|roofline\|detailed]` | Profiles one case under `ncu`, then ingests the report automatically. |
| `kompass ingest <reports/.../report.ncu-rep>` | Parses an existing Nsight Compute report into `records.jsonl` (`profile` calls this for you). |

### Examples

```bash
# What's available?
kompass list

# Benchmark vec_add on the local GPU
kompass bench src/kompass/cases/vec_add_sweep.yaml

# Same sweep on a Modal H100
kompass bench src/kompass/cases/vec_add_sweep.yaml --backend modal:h100

# Profile a single case. The key is <op>/<impl suffix>: `vec_add/triton` -> `vec_add_triton`
kompass profile vec_add/triton -p N=16777216 --dtype float32 --set full
```

> [!NOTE]
> `bench` takes the **full** impl names in the YAML (`vec_add_triton`). `profile` takes
> `op/suffix` (`vec_add/triton`) and joins the two parts with `_`.

### Outputs

```
logs/<timestamp>_<run_id>/
├── meta.json        # run config, GPU/driver/nvcc info, measured copy bandwidth, git commit
├── times.jsonl      # bench: one row per case (times at each quantile, flops, bytes, status, memory)
└── records.jsonl    # profile: one row per kernel launch (duration, DRAM bytes, %DRAM / %SM of peak)

reports/<op>_<impl>/<uid>/
├── meta.json
└── report.ncu-rep   # open in Nsight Compute UI
```

Every row is flat: params appear as `param_<name>` columns. That makes loading a run a
one-liner, e.g. `pd.read_json("logs/.../times.jsonl", lines=True)`.

---

## File tree

```
.
├── Dockerfile                  # CUDA 13 + Nsight Compute + uv toolchain (devcontainer, pods, Modal)
├── docker/
│   ├── start.sh                # pod entrypoint: sshd, repo clone + uv sync, ncu preflight
│   └── ncu-check.sh            # checks that ncu can read GPU perf counters
├── .devcontainer/              # VS Code devcontainer (bind-mounts repo, caches uv/ccache/torch ext)
├── example.env                 # template for .env (Modal credentials)
├── pyproject.toml / uv.lock
├── roofline.ipynb              # roofline plots from Op flops/bytes + measured results
├── vec_add_vis.ipynb           # vec_add benchmark visualisation
├── logs/                       # bench / profile records (generated)
├── reports/                    # ncu reports (generated)
└── src/kompass/
    ├── cli/main.py             # Typer app: list, bench, profile, ingest
    ├── runner.py               # run_bench(): copy-bandwidth probe, per-case memory check, bench loop
    ├── infra/
    │   └── modal_runner.py     # Modal app/image/volumes wrapping run_bench on a remote GPU
    ├── harness/
    │   ├── ops.py              # Op ABC: reference, compare, make_inputs, bytes_moved, flops
    │   ├── loaders.py          # TorchLoader, TritonLoader, CudaLoader (JIT, per-arch cache)
    │   ├── spec.py             # KernelSpec = Op + Loader; test() and bench()
    │   ├── registry.py         # OpEntry(op, impls)
    │   ├── case.py, configs.py # Case and SweepConfig dataclasses
    │   ├── profile.py          # child process launched under ncu (warmup + profiled call)
    │   ├── records.py          # ProfileRecord
    │   └── utils.py            # env description, impl/dtype resolution, git commit, ...
    ├── cases/                  # sweep definitions (*.yaml)
    │   ├── vec_add_sweep.yaml
    │   ├── softmax_occupancy.yaml
    │   ├── softmax_rowlen.yaml
    │   └── matmul_sweep.yaml
    └── kernels/
        ├── __init__.py         # global REGISTRY (merges every Op's registry)
        ├── common.cuh          # CUDA_CHECK / CUDA_CHECK_LAUNCH macros
        ├── vec_add/            # VecAdd Op + CUDA, Triton, Torch impls
        ├── softmax/            # Softmax Op + Triton (single-block, online), Torch impls
        └── matmul/             # Matmul Op + Triton (tiled), Torch impls
```

---

## Setup

**Requirements:** an NVIDIA GPU with driver ≥ 580 (CUDA 13.x) for local runs, or a
[Modal](https://modal.com) account for remote runs. Python 3.14 and [uv](https://docs.astral.sh/uv/).

### Option A: devcontainer (recommended)

1. Create a `.env` file from [example.env](example.env) (the devcontainer passes it to the container):
   ```bash
   cp example.env .env   # then fill in MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
   ```
2. In VS Code, run **Dev Containers: Reopen in Container**. The image includes CUDA, `nvcc`, `ncu`,
   clangd and the Python environment, and `uv sync --frozen` runs on creation.
3. Check that it works:
   ```bash
   kompass list
   ncu-check        # only needed for `kompass profile`
   ```

### Option B: bare metal

```bash
uv sync --frozen
source .venv/bin/activate
kompass list
```

You need `nvcc` on `PATH` for CUDA kernels and `ncu` for profiling.

### Modal (remote GPUs)

```bash
modal token set --token-id $MODAL_TOKEN_ID --token-secret $MODAL_TOKEN_SECRET
modal volume create cuda_kernels     # caches compiled CUDA extensions
modal volume create triton_kernels   # caches Triton compilations
kompass bench src/kompass/cases/vec_add_sweep.yaml --backend modal:t4
```

The Modal image is built from the same `Dockerfile`, so local and remote runs share one
environment. `modal:h100` is pinned to `H100!` so that Modal doesn't silently upgrade it to an H200.

> [!TIP]
> If `ncu-check` reports `ERR_NVGPUCTRPERM`, the host doesn't expose GPU performance counters.
> The container needs `--cap-add SYS_ADMIN`, or the host driver needs
> `NVreg_RestrictProfilingToAdminUsers=0`. Benchmarking still works without either.

---

## Adding a kernel

Every Op lives in its own package under `src/kompass/kernels/<op>/`. The walkthrough below
uses `vec_add`. The same steps apply to any new Op.

```
src/kompass/kernels/vec_add/
├── __init__.py   # Op definition + KernelSpecs + REGISTRY
├── vec_add.py    # Triton kernel + Python launcher
└── vecAdd.cu     # CUDA kernel + pybind11 binding
```

### 1. Define the Op

The Op is the contract that every implementation is tested and measured against:

```python
# kernels/vec_add/__init__.py
class VecAdd(Op):
    param_names: tuple[str, ...] = ("N",)   # must match the axes in the sweep YAML

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

### 2a. Write a Triton kernel

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

### 2b. Write a CUDA kernel

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

### 3. Register the implementations

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
For a new Op, also merge its registry into the global one in
[kernels/\_\_init\_\_.py](src/kompass/kernels/__init__.py):

```python
from .vec_add import REGISTRY as _vec_add
from .my_op import REGISTRY as _my_op

for r in [_vec_add, ..., _my_op]:
    REGISTRY.update(r)
```

### 4. Describe a sweep

```yaml
# src/kompass/cases/vec_add_sweep.yaml
op: vec_add
impls: [vec_add_cuda, vec_add_triton, vec_add_torch]
axes:
  N: [1024, 1048576, 268435456]   # one list per entry in Op.param_names
  dtypes: [float32]
seed: 0
quantiles: [0.2, 0.5, 0.8]        # passed to do_bench
```

### 5. Run it

```bash
kompass list                                              # the new impls should be listed
kompass bench src/kompass/cases/vec_add_sweep.yaml        # correctness-gated benchmark
kompass profile vec_add/cuda -p N=16777216                # Nsight Compute deep dive
```

---

## Tips

- **Correctness first.** A failing `compare` marks the case `incorrect` and skips timing.
  Check the `error` field in `times.jsonl` for the max abs diff.
- **Compilation cache.** CUDA builds are keyed on `sm_XY` plus a hash of the sources, so editing a
  `.cu` file triggers a rebuild. The devcontainer keeps `ccache` and `torch_extensions` on volumes,
  and Modal keeps them on the `cuda_kernels` and `triton_kernels` volumes.
- **Autograd.** Set `supports_backward=True` on a loader to have `make_inputs` create tensors
  with `requires_grad=True`.
- **Roofline.** `meta.json` records the copy bandwidth measured with Triton on the target GPU.
  Plot achieved GB/s (`bytes_moved / time`) against it, or use [roofline.ipynb](roofline.ipynb).

## Roadmap

See [TODO.md](TODO.md): CUDA-graph timing (`do_bench_cudagraph`), explicit arch flags in
`CudaLoader`, and a `tune` endpoint.
