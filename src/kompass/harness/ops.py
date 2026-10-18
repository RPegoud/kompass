from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class CompareResult:
    correct: bool
    max_abs_diff: float


class Op(ABC):
    param_names: tuple[str, ...]
    option_names: tuple[str, ...] | None

    def __init__(self, relative_tolerance: float | None = None) -> None:
        self.relative_tolerance = relative_tolerance

    @property
    def name(self) -> str:
        return type(self).__name__.lower()

    @staticmethod
    @abstractmethod
    def reference(
        *tensors: torch.Tensor,
        params: dict[str, int],
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compare(
        self,
        pred: torch.Tensor,
        ref: torch.Tensor,
    ) -> CompareResult:
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
    def flops(params: dict[str, int]) -> int:
        raise NotImplementedError
