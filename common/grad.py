import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import *

import numpy as np

from typing import Any, Tuple, Union
from numbers import Number

class Grad(metaclass=ABCMeta): 
    def __init__(self, *inputs) -> None:
        self._inputs = inputs
    
    @abstractmethod
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.backward(*args, **kwds)
    
    def is_leaf(self) -> bool: 
        return len(self._inputs) == 0

class IdentityGrad(Grad):
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (grad, )

class AddGrad(Grad): 
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]: 
        return (grad, ) * len(self._inputs)

class ReshapeGrad(Grad): 
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]: 
        return (grad.reshape(*self._inputs.shape), )

class TransposeGrad(Grad):
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]: 
        return (grad.T, )

class ScalarMulGrad(Grad): # order: (scalar) * (array)
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]: 
        return (grad * self._inputs[1]._array, grad * self._inputs[0]._array)

class MatMulGrad(Grad): 
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (grad @ self._inputs[1]._array.T, self._inputs[0]._array.T @ grad)

class ExpandGrad(Grad): 
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]: 
        return (np.mean(grad, axis=0), )

class SumGrad(Grad): 
    def backward(self, grad: Union[float, np.ndarray]) -> Tuple[np.ndarray]: # arr
        if isinstance(grad, Number): 
            grad = np.array(grad, dtype=np.float64)
        reps = (a // b for a, b in zip(self._inputs[0].shape, grad.shape))
        return (np.tile(grad, reps), )

class PowerGrad(Grad): 
    def __init__(self, input_, exp) -> None:
        super().__init__(input_)
        self._exp = exp
    
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (grad * self._exp * np.power(self._inputs[0]._array, self._exp - 1), )

class ExpGrad(Grad): 
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]: 
        return (grad * np.exp(self._inputs[0]._array), )
