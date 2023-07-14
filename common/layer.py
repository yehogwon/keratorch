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
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.x = x
        return self.W @ x + np.tile(self.b, reps=(self.x.shape[0], 1, 1))

    def backward(self, grad: float, optimizer: Callable) -> float: 
        # FIXME: check if it works properly in batched setting
        self.w_grad = grad @ self.x.T
        self.b_grad = grad
        self.x_grad = self.W.T @ grad
        self.W = optimizer(self.W, self.w_grad)
        self.b = optimizer(self.b, self.b_grad)
        return self.x_grad

class Convolution(Layer): 
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, padding: int=0) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.b = np.random.randn(out_channels)

    def forward(self, x: np.ndarray) -> np.ndarray: 
        if x.ndim == 3:
            x = x.reshape(1, *x.shape)
        # TODO: implement forward pass of convolution
    
    def backward(self, grad: float, optimizer: Callable) -> float: 
        # TODO: implement backward pass of convolution
        pass
