from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class TestResult:
    correct: bool
    max_abs_diff: float


class Op(ABC):
    param_names: tuple[str, ...]

    @property
    def name(self) -> str:
        return type(self).__name__

    @staticmethod
    @abstractmethod
    def reference(*tensors: torch.Tensor, params: dict[str, int]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compare(self, pred: torch.Tensor, ref: torch.Tensor) -> TestResult:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def make_inputs(
        params: dict[str, int],
        dtype: torch.dtype,
        requires_grad: bool,
        seed: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def bytes_moved(params: dict[str, int], dtype: torch.dtype) -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def flops(params: dict[str, int], dtype: torch.dtype) -> int:
        raise NotImplementedError
