from abc import *
import numpy as np

from typing import Any, Callable

class Loss(metaclass=ABCMeta): 
    def __init__(self) -> None:
        pass

    @abstractmethod
    def forward(self, y: np.ndarray, gt: np.ndarray) -> np.ndarray: 
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray, optimizer: Callable) -> float: 
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

class MSE(Loss): 
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y: np.ndarray, gt: np.ndarray) -> np.ndarray: 
        return np.mean(np.sum((y - gt) ** 2, axis=1))

    def backward(self, grad: np.ndarray, optimizer: Callable) -> float: 
        return grad * 2 / np.prod(grad.shape)
