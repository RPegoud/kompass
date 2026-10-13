import hashlib
import importlib
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.cpp_extension import load


class Loader(Protocol):
    fn_name: str
    supports_backward: bool

    def load(self) -> Callable:
        raise NotImplementedError


class TorchLoader(Loader):
    def __init__(self, fn: Callable, options: dict[str, int | str] | None = None):
        self.fn_name = f"torch_{fn.__name__}"
        self.supports_backward = True
        if options:
            self.fn = partial(fn, **options)
        else:
            self.fn = fn

    def load(self) -> Callable:
        return self.fn


class TritonLoader(Loader):
    def __init__(
        self,
        module_path: str,
        fn_name: str,
        supports_backward=False,
    ) -> None:
        self.module_path = module_path
        self.fn_name = fn_name
        self.supports_backward = supports_backward
        self.backend = "triton"
        self.module = None

    def load(self) -> Callable:
        if self.module is None:
            self.module = importlib.import_module(self.module_path)
            self.kernel: Callable = getattr(self.module, self.fn_name)
        return self.kernel


class CudaLoader(Loader):
    def __init__(
        self,
        module_name: str,  # sets `TORCH_EXTENSION_NAME`, arch specific
        fn_name: str,  # python-visible fn name
        sources: str | list[str],
        supports_backward: bool = False,
    ) -> None:
        self.module_name = module_name
        self.fn_name = fn_name
        self.sources = sources
        self.supports_backward = supports_backward
        self.backend = "cuda"
        self.module = None

    def load(self) -> Callable:
        if self.module is None:
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}.{minor}"
            src_hash = hashlib.sha256(
                b"".join(open(s, "rb").read() for s in self.sources)  # noqa: SIM115
            ).hexdigest()[:8]
            src_paths = [Path(s).resolve() for s in self.sources]
            include_dirs = sorted({str(p.parent.parent) for p in src_paths})

            os.environ["TORCH_CUDA_ARCH_LIST"] = arch  # TODO: move to container entry

            self.module = load(
                name=f"{self.module_name}_sm{major}{minor}_{src_hash}",  # compile and cache once per gpu
                sources=self.sources,
                extra_include_paths=include_dirs,
                with_cuda=True,
                verbose=True,
            )
        self.kernel: Callable = getattr(self.module, self.fn_name)
        return self.kernel
