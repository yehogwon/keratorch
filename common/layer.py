import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import *
import numpy as np

from typing import Any, Callable, List

from common.array import GradArray, expand
from util.matrix import cross_correlate, convolve

class Layer(metaclass=ABCMeta): 
    def __init__(self) -> None:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: 
        pass

    def get_params(self) -> List[GradArray]: 
        params = []
        for _, v in self.__dict__.items():
            if isinstance(v, GradArray):
                params.append(v)
            elif isinstance(v, Layer):
                params += v.get_params()
        return params

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

class Linear(Layer): 
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.W = GradArray(np.random.randn(output_dim, input_dim), name='weight')
        self.b = GradArray(np.random.randn(output_dim), name='bias')

    def forward(self, x: GradArray) -> GradArray: 
        # return GradArray(x) @ self.W.T  + self.b
        new_b = self.b if x.n_dim == 1 else expand(self.b, x.shape[0])
        return x @ self.W.T + new_b # Do not use broadcasting since it is too complicated to implement by my own

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
