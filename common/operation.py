from abc import *
import numpy as np

from typing import Tuple, Union
from numbers import Number

from util.matrix import cross_correlate

class Operation(metaclsss=ABCMeta): 
    @abstractmethod
    def forward(self) -> np.ndarray: 
        pass

    @abstractmethod
    def backward(self, grad: Union[float, np.ndarray]) -> Tuple[np.ndarray]:
        # returns a tuple of gradients. 
        # ret[j]: gradient w.r.t. the j-th input
        pass

class Identity(Operation): 
    def __init__(self) -> None:
        self.x = None

    def forward(self, x: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]: 
        self.x = x
        return x

    def backward(self, grad: Union[float, np.ndarray]) -> Union[float, np.ndarray]: 
        return grad

class Add(Operation): 
    def __init__(self) -> None:
        self.x = None
        self.y = None

    def forward(self, x: Union[Number, np.ndarray], y: Union[Number, np.ndarray]) -> Union[Number, np.ndarray]: 
        self.x = x
        self.y = y
        return self.x + self.y
    
    def backward(self, grad: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray]]: 
        return grad, grad

class ScalarMul(Operation): 
    def __init__(self) -> None:
        self.c = None
        self.x = None

    def forward(self, c: float, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]: 
        self.c = c
        self.x = x
        return self.c * self.x

    def backward(self, grad: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray]]: 
        return self.c * grad, np.sum(self.x * grad)

class MatMul(Operation): 
    def __init__(self) -> None:
        self.A = None
        self.B = None

    def forward(self, A: np.ndarray, B: np.ndarray) -> np.ndarray: 
        self.A = A
        self.B = B
        return self.A @ self.B

    def backward(self, grad: np.ndarray) -> np.ndarray: 
        return grad @ self.B.T, self.A.T @ grad

class CrossCorrelate(Operation): 
    def __init__(self) -> None: 
        super().__init__()
        self.X = None
        self.W = None
        self.stride = None
        self.padding = None
    
    def forward(self, X: np.ndarray, W: np.ndarray, stride: int, padding: int) -> np.ndarray: 
        self.X = X
        self.W = W
        self.stride = stride
        self.padding = padding
        return cross_correlate(X, W, stride, padding)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]: 
        raise NotImplementedError('CrossCorrelate.backward() is not implemented')

class Convolve(Operation): 
    def __init__(self) -> None: 
        super().__init__()
        self.X = None
        self.W = None
        self.stride = None
        self.padding = None
    
    def forward(self, X: np.ndarray, W: np.ndarray, stride: int, padding: int) -> np.ndarray: 
        self.X = X
        self.W = W
        self.stride = stride
        self.padding = padding
        return cross_correlate(X, np.flip(W), stride, padding)

    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]: 
        raise NotImplementedError('Convolve.backward() is not implemented')
