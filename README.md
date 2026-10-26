# `Kompass`

<a href="Kompass.png?raw=true"><img src="Kompass.png?raw=true" width="200" align="right" /></a>

**Kompass is a tooling suite for GPU kernel development.** It gives you one harness to
**test, benchmark, tune and profile** CUDA, Triton and PyTorch implementations of the same
operation side by side, and to turn the results into a report. Runs can happen on your local GPU or on a cloud GPU through
[Modal](https://modal.com).

You describe an **Op** once (how to build inputs, a reference result, a correctness check,
and FLOP/byte counts). You then register any number of **implementations** of it. 

Kompass:
- **JIT-compiles** CUDA extensions per GPU architecture and source hash, and imports Triton kernels lazily.
- **Gates every benchmark on correctness**: a kernel whose output doesn't match the reference is never timed.
- **Sweeps** parameter grids from a YAML file and records each case's status (`ok`, `incorrect`, `oom`, `skipped_oom`, `failed`).
- **Measures copy bandwidth** on the target GPU, so results can be compared against an achievable roofline.
- **Tunes** each kernel's launch parameters (block sizes, warps, stages, ...) per GPU and problem shape, and
  benchmarks with the best config found, or with and without it side by side.
- **Reports** each sweep as plots (throughput, % of ceiling, % of baseline, roofline, tuning gain) and a results
  table, judging every case against the vendor peak of the arithmetic it runs.
- **Profiles** a single case under Nsight Compute and turns the `.ncu-rep` into flat JSONL records.
- Stamps every run with the git commit (`-dirty` if there are uncommitted changes) and the GPU, driver and CUDA versions, so results stay reproducible.

---

## ⌨️ CLI endpoints

After `uv sync`, all endpoints are available through the `kompass` entry point.

| Command | What it does |
| --- | --- |
| `kompass list` | Prints every registered Op and its implementations. |
| `kompass runs [--op <op>] [--config <config_name>] [--n 20]` | Lists the most recent runs in `logs/`. |
| `kompass bench <config_name> [<config_name> ...] [--backend local\|modal:<gpu>] [--device cuda:0] [--tuning tuned\|untuned\|both]` | Runs one or more sweeps: each `impl × dtype × parameter set` in the YAML (from an `axes` grid or an explicit `shapes` list, see [Describe a sweep](#-4-describe-a-sweep)) is checked for correctness, then timed with `triton.testing.do_bench`. Sweeps run one after another locally, and in parallel on Modal (one GPU each). A failed sweep doesn't stop the others; `bench` exits with code 1 if any failed. `--tuning` (default `tuned`) uses the configs saved by `tune` for the GPU it runs on; `both` also runs each tuned case with the launcher defaults, in the same run. |
| `kompass tune <config_name> [<config_name> ...] [--backend local\|modal:<gpu>]` | Searches each tunable impl's `search_space` on the sweep's cases (see [Make it tunable](#-6-make-it-tunable)) and saves the fastest config per case to `src/kompass/tuning/<gpu>/<impl>.json`. Each config is timed briefly, then the 5 fastest per case are re-timed in full on the same GPU. On Modal, every (impl, precision, shape) group is its own input, spread over up to `max_containers` GPUs, costliest first. |
| `kompass profile <op>/<impl> -p K=V [-p K=V ...] [-o K=V ...] [--dtype float32] [--set full\|roofline\|detailed]` | Profiles one case under `ncu`, then ingests the report automatically. `-o` sets kernel options, e.g. `-o input_precision=tf32`. |
| `kompass ingest <reports/.../report.ncu-rep>` | Parses an existing Nsight Compute report into `records.jsonl` (`profile` calls this for you). |
| `kompass report [--config <config_name>] [--run-id <id>] [--gpu <name>] [--baseline <impl>]` | Builds `reports/<config_name>/`: `summary.md` (results table + embedded plots), `summary.csv`, `plots/*.png`, and every plot in one multi-page `report.pdf`. Uses the newest `bench` run of each sweep, one section per sweep (op + config), so sweeps of the same op never share an axis; each baseline defaults to the op's `torch` impl. Arithmetics that share a dtype (e.g. `float32` and `tf32`) get side-by-side panels, and tuned runs appear as `<impl> (tuned)` with a tuning-gain plot. |

### 💡 Examples

```bash
# What's available?
kompass list

# What are the 10 most recent logs for experiment `matmul_k_sweep`?  
kompass runs --op "matmul" --config "matmul_k_sweep" --n 10

# Benchmark vec_add on the local GPU
kompass bench vec_add_sweep

# Same sweep on a Modal H100
kompass bench vec_add_sweep --backend modal:h100

# Two sweeps in parallel, each on its own Modal H100
kompass bench vec_add_sweep matmul_square --backend modal:h100

# Tune the matmul kernels, then compare tuned and default configs in one run
kompass tune matmul_square matmul_k_sweep --backend modal:h100
kompass bench matmul_square --backend modal:h100 --tuning both
kompass report --config matmul_square

# Profile a single case. The key is <op>/<impl suffix>: `vec_add/triton` -> `vec_add_triton`
kompass profile vec_add/triton -p N=16777216 --dtype float32 --set full
```

> [!NOTE]
> `bench` takes the **full** impl names in the YAML (`vec_add_triton`). `profile` takes
> `op/suffix` (`vec_add/triton`) and joins the two parts with `_`.

### 📂 Outputs

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

---

## 🌳 File tree

```
.
├── Dockerfile                  # CUDA 13 + Nsight Compute + uv toolchain (devcontainer, pods, Modal)
├── docker/
│   ├── start.sh                # pod entrypoint: sshd, repo clone + uv sync, ncu preflight
│   └── ncu-check.sh            # checks that ncu can read GPU perf counters
├── .devcontainer/              # VS Code devcontainer (bind-mounts repo, caches uv/ccache/torch ext)
├── example.env                 # template for .env (Modal credentials)
├── pyproject.toml / uv.lock
├── logs/                       # bench / profile records (generated)
├── reports/                    # ncu reports (generated)
└── src/kompass/
    ├── cli/main.py             # Typer app: list, runs, bench, tune, profile, ingest, report
    ├── runner.py               # run_bench(), run_tune() (two-pass search with early abort), run_local()
    ├── infra/
    │   ├── modal_cfg.yaml      # volumes, max_containers, timeout
    │   └── modal_runner.py     # Modal app/image/volumes; run_modal() dispatches payloads in parallel, with progress
    ├── harness/
    │   ├── ops.py              # Op ABC: reference, compare, make_inputs, bytes_moved, flops
    │   ├── loaders.py          # TorchLoader, TritonLoader, CudaLoader (JIT, per-arch cache)
    │   ├── spec.py             # KernelSpec = Op + Loader; test() and bench()
    │   ├── registry.py         # OpEntry(op, impls)
    │   ├── schema.py           # plain dataclasses: SweepConfig, Case, Payload, CaseResult, ProfileRecord
    │   ├── tuning.py           # save / load / apply the configs found by `tune`
    │   ├── report.py           # records -> plots, tables, PDF; vendor peaks per GPU
    │   ├── profile.py          # child process launched under ncu (warmup + profiled call)
    │   └── utils.py            # env description, impl/dtype resolution, git commit, ...
    ├── tuning/<gpu>/<impl>.json # best configs found by `tune` (commit them)
    ├── cases/                  # sweep definitions (*.yaml)
    │   ├── vec_add_sweep.yaml
    │   ├── softmax_occupancy.yaml
    │   ├── softmax_rowlen.yaml
    │   ├── matmul_k_sweep.yaml
    │   └── matmul_square.yaml
    └── kernels/
        ├── __init__.py         # global REGISTRY (merges every Op's registry)
        ├── common.cuh          # CUDA_CHECK / CUDA_CHECK_LAUNCH macros
        ├── <op_name>/
            └── <op_name>.cu    # define CUDA kernels here
            └── <op_name>.py    # define Triton kernels here
```

---

## 🛠️ Setup

**Requirements:** an NVIDIA GPU with driver ≥ 580 (CUDA 13.x) for local runs, or a
[Modal](https://modal.com) account for remote runs. Python 3.14 and [uv](https://docs.astral.sh/uv/).

### 🐳 Option A: devcontainer (recommended)

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

### 🖥️ Option B: bare metal

```bash
uv sync --frozen
source .venv/bin/activate
kompass list
```

You need `nvcc` on `PATH` for CUDA kernels and `ncu` for profiling.

### ☁️ Modal (remote GPUs)

```bash
modal token set --token-id $MODAL_TOKEN_ID --token-secret $MODAL_TOKEN_SECRET
modal volume create cuda_kernels     # caches compiled CUDA extensions
modal volume create triton_kernels   # caches Triton compilations
kompass bench vec_add_sweep --backend modal:t4
```

The Modal image is built from the same `Dockerfile`, so local and remote runs share one
environment. `modal:h100` is pinned to `H100!` so that Modal doesn't silently upgrade it to an H200.
Other requests can still be upgraded (e.g. `A100-40GB` may run on an 80GB card): records and tuned
configs are keyed on the device name the container reports, never on the requested GPU.

[infra/modal_cfg.yaml](src/kompass/infra/modal_cfg.yaml) sets the cache volumes, `max_containers`
(the number of GPUs used at once, 10 on Modal's free plan) and the per-call `timeout`.

> [!TIP]
> If `ncu-check` reports `ERR_NVGPUCTRPERM`, the host doesn't expose GPU performance counters.
> The container needs `--cap-add SYS_ADMIN`, or the host driver needs
> `NVreg_RestrictProfilingToAdminUsers=0`. Benchmarking still works without either.

---

## 🧩 Adding a kernel

Every Op lives in its own package under `src/kompass/kernels/<op>/`. The walkthrough below
uses `vec_add`. The same steps apply to any new Op.

```
src/kompass/kernels/vec_add/
├── __init__.py   # Op definition + KernelSpecs + REGISTRY
├── vec_add.py    # Triton kernel + Python launcher
└── vecAdd.cu     # CUDA kernel + pybind11 binding
```

### 📐 1. Define the Op

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

### 🔱 2a. Write a Triton kernel

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

### ⚡ 2b. Write a CUDA kernel

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

### 📝 3. Register the implementations

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
[kernels/\_\_init\_\_.py](src/kompass/kernels/__init__.py):

```python
from .vec_add import REGISTRY as _vec_add
from .my_op import REGISTRY as _my_op

for r in [_vec_add, ..., _my_op]:
    REGISTRY.update(r)
```

### 🧪 4. Describe a sweep

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

### 🚀 5. Run it

```bash
kompass list                                              # the new impls should be listed
kompass bench vec_add_sweep                               # correctness-gated benchmark
kompass profile vec_add/cuda -p N=16777216                # Nsight Compute deep dive
```

### 🔧 6. Make it tunable

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

---

## 💬 Tips

- **Correctness first.** A failing `compare` marks the case `incorrect` and skips timing.
  Check the `error` field in `times.jsonl` for the max abs diff.
- **Compilation cache.** CUDA builds are keyed on `sm_XY` plus a hash of the sources, so editing a
  `.cu` file triggers a rebuild. The devcontainer keeps `ccache` and `torch_extensions` on volumes,
  and Modal keeps them on the `cuda_kernels` and `triton_kernels` volumes.
- **Autograd.** Set `supports_backward=True` on a loader to have `make_inputs` create tensors
  with `requires_grad=True`.

## 🗺️ Roadmap

- [ ] Add support for:
  - [ ] Torch Inductor (`torch.compile()`)
  - [ ] CuTe DSL
  - [ ] JAX Jitted scripts
