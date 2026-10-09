#!/usr/bin/env bash
# Checks that Nsight Compute can read GPU performance counters in this container
# by profiling a single tiny torch kernel. Exit 0 = `kompass profile` will work.
set -uo pipefail

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[ncu-check] no GPU visible (nvidia-smi failed)"
    exit 1
fi

out=$(ncu --metrics gpu__time_duration.sum \
    python -c "import torch; torch.ones(1, device='cuda').add_(1); torch.cuda.synchronize()" 2>&1)

if grep -q ERR_NVGPUCTRPERM <<<"$out"; then
    cat <<'EOF'
[ncu-check] FAIL: ERR_NVGPUCTRPERM, no access to GPU performance counters.
  This is a host setting, not something the image can fix. It needs either
  the host driver option NVreg_RestrictProfilingToAdminUsers=0, or the
  container started with --cap-add SYS_ADMIN. If you can't change either on
  this host, use a different host/provider for profiling (benchmarking still works).
EOF
    exit 1
elif grep -q gpu__time_duration.sum <<<"$out"; then
    echo "[ncu-check] OK: Nsight Compute can profile on $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
else
    echo "[ncu-check] FAIL: unexpected ncu output:"
    tail -n 20 <<<"$out"
    exit 1
fi
