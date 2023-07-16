from abc import *
import numpy as np
from typing import Any, Union, Tuple, List

from array import GradArray

class Optimizer(metaclass=ABCMeta): 
    def __init__(self, *params: Union[Tuple[GradArray], List[GradArray]]) -> None:
        self._params = params
        
    @abstractmethod
    def step(self) -> None: 
        pass
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.step(*args, **kwds)

class SGD(Optimizer): 
    def __init__(self, *params: Union[Tuple[GradArray], List[GradArray]],  lr: float) -> None:
        super().__init__(*params)
        self._lr = lr
    
    def step(self) -> None: 
        for param in self._params:
            param._array -= param.grad * self._lr
