from abc import *
import numpy as np

from typing import Any, Union
from array import GradArray, sum, l2_norm_square

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
        self.out = sum(l2_norm_square(y - gt)) / y.shape[0]

    def backward(self, grad: float=1) -> GradArray: 
        self.out.backward(grad)
        return self.y.grad
