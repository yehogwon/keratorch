from abc import *

import numpy as np

from typing import Any, Tuple

class Grad(metaclass=ABCMeta): 
    def __init__(self, *inputs) -> None:
        self._inputs = inputs
    
    @abstractmethod
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.backward(*args, **kwds)

class IdentityGrad(Grad):
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (grad, )

class AddGrad(Grad): 
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]: 
        return (grad, ) * len(self._inputs)

class ReshapeGrad(Grad): 
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]: 
        return (grad.reshape(*self._inputs.shape), )

class ScalarMulGrad(Grad): # order: (scalar) * (array)
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]: 
        return (grad * self._inputs[1]._array, grad * self._inputs[0]._array)

class MatMulGrad(Grad): 
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (grad @ self._inputs[1]._array.T, self._inputs[0]._array.T @ grad)
