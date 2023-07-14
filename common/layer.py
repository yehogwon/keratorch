from abc import *
import numpy as np

from typing import Any, Callable

from util.matrix import cross_correlate, convolve

class Layer(metaclass=ABCMeta): 
    def __init__(self) -> None:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: 
        pass

    @abstractmethod
    def backward(self, grad: Any, optimizer: Callable) -> Any: 
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
        # return self.W @ x + np.tile(self.b, reps=(self.x.shape[0], 1, 1))
        return self.W @ x + self.b # Thank you, broadcasting!

    def backward(self, grad: np.ndarray, optimizer: Callable) -> np.ndarray: 
        self.w_grad = self.x.T @ grad
        self.b_grad = grad
        self.x_grad = grad @ self.W.T
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
        self.x = x
        return cross_correlate(x, self.W, self.stride, self.padding) + np.tile(self.b.reshape(self.out_channels, 1, 1), reps=[self.x.shape[0], 1, 1, 1])
    
    def backward(self, grad: np.ndarray, optimizer: Callable) -> np.ndarray: 
        # TODO: implement backward pass of convolution
        pass
