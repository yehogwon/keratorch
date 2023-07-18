import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import *
import numpy as np

from typing import List, Tuple

from common.array import GradArray
from common.layer import Layer
from common.grad import Grad

from util import F

class Activation(Layer, metaclass=ABCMeta): 
    @abstractmethod
    def op(self, x: GradArray) -> GradArray:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray: 
        self.x = x
        self.out = self.op(x)
        return self.out

    def get_params(self) -> List[GradArray]:
        return []

class Sigmoid(Activation): 
    def op(self, x: GradArray) -> GradArray: 
        return GradArray(F.sigmoid(x._array), grad_op=SigmoidGrad(x), name='sigmoid')

class SigmoidGrad(Grad): 
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]: 
        return (grad * F.sigmoid(self._inputs[0]._array) * (1 - F.sigmoid(self._inputs[0]._array)), )

class Tanh(Activation): 
    def op(self, x: GradArray) -> GradArray: 
        return GradArray(F.tanh(x._array), grad_op=TanhGrad(x), name='tanh')

class TanhGrad(Grad): 
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        return (grad * (1 - F.tanh(self._inputs[0]._array) ** 2), )

class ReLU(Activation): 
    def op(sefl, x: GradArray) -> GradArray: 
        return GradArray(F.relu(x._array), grad_op=ReLUGrad(x), name='relu')

class ReLUGrad(Grad): 
    def backward(self, grad: np.ndarray) -> Tuple[np.ndarray]: 
        return (grad * (self._inputs[0]._array > 0), )
