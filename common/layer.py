from abc import *
import numpy as np

from typing import Any, Callable

class Layer(metaclass=ABCMeta): 
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

class Linear(Layer): 
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.W = np.random.randn(output_dim, input_dim)
        self.b = np.random.randn(output_dim, 1)

    def forward(self, x: np.ndarray) -> np.ndarray: 
        self.x = x
        return self.W @ x + self.b

    def backward(self, grad: float, optimizer: Callable) -> float: 
        self.w_grad = grad @ self.x.T
        self.b_grad = grad
        self.x_grad = self.W.T @ grad
        self.W = optimizer(self.W, self.w_grad)
        self.b = optimizer(self.b, self.b_grad)
        return self.x_grad
