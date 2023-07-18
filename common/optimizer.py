import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import *
from typing import Any, Union, Tuple, List, Dict

from common.array import GradArray
from util import F

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

class Adam(Optimizer): 
    ms: Dict[int, float]
    vs: Dict[int, float]

    def __init__(self, params: List[GradArray], lr: float=0.001, beta1: float=0.9, beta2: float=0.999, eps: float=1e-8) -> None:
        super().__init__(*params)
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self.ms = {}
        self.vs = {}

    def step(self) -> None:
        for param in self._params:
            if id(param) not in self.ms:
                self.ms[id(param)] = 0
                self.vs[id(param)] = 0
            
            self.ms[id(param)] = self._beta1 * self.ms[id(param)] + (1 - self._beta1) * param._grad
            self.vs[id(param)] = self._beta2 * self.vs[id(param)] + (1 - self._beta2) * F.l2_norm(param._grad).item() ** 2
            m_hat = self.ms[id(param)] / (1 - self._beta1)
            v_hat = self.vs[id(param)] / (1 - self._beta2)
            
            param._array -= self._lr * m_hat / (v_hat ** 0.5 + self._eps)
