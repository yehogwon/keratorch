import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import *
from typing import Any, Union, Tuple, List

from common.array import GradArray

class Optimizer(metaclass=ABCMeta): 
    def __init__(self, *params: Union[Tuple[GradArray], List[GradArray]]) -> None:
        self._params = params
        
    @abstractmethod
    def step(self) -> None: 
        pass
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.step(*args, **kwds)

class SGD(Optimizer): 
    def __init__(self, params: List[GradArray],  lr: float) -> None:
        super().__init__(*params)
        self._lr = lr
    
    def step(self) -> None: 
        for param in self._params:
            param._array -= param._grad * self._lr
