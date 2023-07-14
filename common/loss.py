from abc import *
import numpy as np

from typing import Any, Callable

class Activation(metaclass=ABCMeta): 
    def __init__(self) -> None:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: 
        pass

    @abstractmethod
    def backward(self, grad: float, optimizer: Callable) -> float: 
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
