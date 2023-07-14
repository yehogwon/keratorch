from abc import *
import numpy as np
from typing import Union, List

from layer import Layer
from activation import Activation

class Optimizer(metaclass=ABCMeta): 
    def __init__(self) -> None:
        pass
        
    @abstractmethod
    def step(self, x: np.ndarray, grad: float) -> None: 
        pass
