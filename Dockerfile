# Kompass toolchain image: CUDA + Nsight Compute + Python deps. Used both as the
# local devcontainer and as a Runpod/remote pod image. The image ships the
# *environment* only; the repo lives on /workspace (bind mount locally, network
# volume + git clone/rsync on a pod), see docker/start.sh.
#
# Cloud GPUs are x86_64, so build for amd64 when pushing from an Apple Silicon Mac:
#   docker buildx build --platform linux/amd64 -t <registry>/kompass:latest --push .
#
# Host requirement: NVIDIA driver >= 580 (CUDA 13.x). On Runpod, filter by CUDA >= 13.
FROM nvidia/cuda:13.3.0-devel-ubuntu26.04

ENV DEBIAN_FRONTEND=noninteractive

# openssh-server + rsync: remote access and code sync on pods.
# Host keys are generated per container by start.sh, never baked into the image.
RUN apt-get update && apt-get install -y \
    python3 python3-dev \
    git curl build-essential \
    clang clangd \
    cmake ninja-build ccache \
    openssh-server rsync \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/ssh/ssh_host_* \
    && mkdir -p /run/sshd \
    && git config --system --add safe.directory '*'

COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /uvx /usr/local/bin/

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Build the venv on the distro python 3.14 so the site-packages paths below are stable.
ENV UV_PYTHON_DOWNLOADS=never
# The uv cache and /opt/venv sit on different filesystems (cache mount / volume).
ENV UV_LINK_MODE=copy

# Dependencies only. This layer (torch, triton, ...) is rebuilt when the
# lockfile changes, not on every source edit. The cache mount keeps uv's
# download cache out of the image. The project itself is installed (editable)
# from /workspace at runtime by `uv sync`.
WORKDIR /opt/kompass-lock
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# libtorch ships inside the pip `torch` wheel (lib/, include/, share/cmake/Torch).
# Point CMake at it so `find_package(Torch)` resolves without a separate download.
ENV TORCH_INSTALL_PREFIX=/opt/venv/lib/python3.14/site-packages/torch
ENV CMAKE_PREFIX_PATH="${TORCH_INSTALL_PREFIX}/share/cmake"
ENV LD_LIBRARY_PATH="${TORCH_INSTALL_PREFIX}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Torch's Caffe2 CMake injects a default gencode list containing compute_50,
# which CUDA 13 no longer accepts ("nvcc fatal: Unsupported gpu architecture").
# CudaLoader overrides this with the live device's arch at JIT time; this value
# is only the fallback for builds without a visible GPU.
ENV TORCH_CUDA_ARCH_LIST=8.0

# Cache JIT-compiled extensions and object files on persistent volumes.
# torch.utils.cpp_extension prepends `ccache` automatically when it is on PATH.
ENV TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions
ENV CCACHE_DIR=/root/.cache/ccache
ENV TRITON_CACHE_DIR=/root/.cache/triton_cache

# clangd (clang 21) predates CUDA 13 and cannot parse its headers as-is.
# clang's CUDA wrapper still includes texture_fetch_functions.h, which CUDA 13
# deleted along with the legacy texture-reference API, so stub it out. This is
# editor-only; nvcc never sees it. .clangd carries the matching -I and the
# second half of the workaround (-D_NV_RSQRT_SPECIFIER=).
RUN mkdir -p /opt/clangd-cuda-shim \
    && printf '#pragma once\n' > /opt/clangd-cuda-shim/texture_fetch_functions.h

# SSH sessions don't inherit the container's ENV (PATH, CUDA, TORCH_* ...).
# start.sh snapshots it to /etc/kompass-env.sh; source it before .bashrc's
# interactive guard so `ssh pod kompass ...` works too.
RUN { echo '[ -f /etc/kompass-env.sh ] && . /etc/kompass-env.sh'; cat /root/.bashrc; } > /tmp/bashrc \
    && mv /tmp/bashrc /root/.bashrc

COPY --chmod=755 docker/start.sh /usr/local/bin/kompass-start
COPY --chmod=755 docker/ncu-check.sh /usr/local/bin/ncu-check

WORKDIR /workspace
EXPOSE 22
# The base image's ENTRYPOINT (nvidia_entrypoint.sh) execs this. Devcontainers
# override the command, so this only runs on pods / `docker run`.
CMD ["/usr/local/bin/kompass-start"]
