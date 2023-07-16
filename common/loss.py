import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import *
import numpy as np

from typing import Any, Union
from common.array import GradArray, sum, l2_norm_square

class Loss(metaclass=ABCMeta): 
    def __init__(self) -> None:
        pass

    @abstractmethod
    def forward(self, y: GradArray, gt: GradArray) -> float: 
        pass

    @abstractmethod
    def backward(self, grad: float=1) -> GradArray: 
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

class MSE(Loss): 
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y: GradArray, gt: GradArray) -> float: 
        self.y = y
        self.gt = gt
        self.out = sum(l2_norm_square(y - gt, axis=1), axis=0) / y.shape[0]
        return self.out

    def backward(self, grad: float=1) -> GradArray: 
        self.out.backward(grad)
        return self.y._grad
