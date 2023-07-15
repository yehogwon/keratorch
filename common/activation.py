from abc import *
import numpy as np

from typing import Any, Callable

from layer import Layer

class Sigmoid(Layer): 
    def op(self, x: np.ndarray) -> np.ndarray: 
        return 1 / (1 + np.exp(-x))

    def forward(self, x: np.ndarray) -> np.ndarray: 
        self.x = x
        return self.op(self.x)
    
    def backward(self, grad: float, optimizer: Callable=None) -> float: 
        return grad * self.op(self.x) * (1 - self.op(self.x))

class tanh(Layer): 
    def op(self, x: np.ndarray) -> np.ndarray: 
        return np.tanh(x)

    def forward(self, x: np.ndarray) -> np.ndarray: 
        self.x = x
        return self.op(self.x)
    
    def backward(self, grad: float, optimizer: Callable=None) -> float: 
        return grad * (1 - self.op(self.x) ** 2)
    
class ReLU(Layer):
    def op(self, x: np.ndarray) -> np.ndarray: 
        return np.maximum(0, x)

    def forward(self, x: np.ndarray) -> np.ndarray: 
        self.x = x
        return self.op(self.x)
    
    def backward(self, grad: float, optimizer: Callable=None) -> float: 
        return grad * (self.x > 0).astype(float)
