import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import *
import numpy as np

from typing import Any, Callable, List

from common.array import GradArray, exp
from common.layer import Layer

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
        return 1 / (1 + exp(-x))

class Tanh(Activation): 
    def op(self, x: GradArray) -> GradArray: 
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
