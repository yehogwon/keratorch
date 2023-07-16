import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import *

import numpy as np

from typing import Any, Tuple

# TODO: other functionalities such as indexing, slicing, ...
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
        return (np.mean(grad, axis=0), None)
