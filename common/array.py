import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import *

import numpy as np

from typing import Optional, Union, Any, Tuple, List
from numbers import Number

from common.grad import *

# FIRE: What if :: A = A + B
class GradArray: 
    def __init__(self, array: np.ndarray, grad: Optional[np.ndarray]=None, grad_op: Optional[Grad]=None, name: str='') -> None:
        self._array: np.ndarray = array
        self._grad: np.ndarray = grad
        self._grad_op: Grad = grad_op
        self._name: str = name

        if self._name == '': 
            try: 
                self._name = self.item()
            except: 
                if self.n_dim == 1: 
                    self._name = 'vector'
                elif self.n_dim == 2: 
                    self._name = 'matrix'
                else: 
                    self._name = 'tensor'
    
    def backward(self, grad: np.ndarray) -> None:
        self._grad = grad
        if self._grad_op and not self._grad_op.is_leaf(): 
            upstream_grads = self._grad_op(self._grad)
            for i, upstream_grad in enumerate(upstream_grads):
                prev_input = self._grad_op._inputs[i]
                if isinstance(prev_input, GradArray): 
                    prev_input.backward(upstream_grad)
    
    def reshape(self, *args: Any, **kwargs: Any) -> 'GradArray':
        _array_, _grad_ = self._array.copy(), self._grad.copy()
        new_ = GradArray(_array_.reshape(*args, **kwargs), _grad_.reshape(*args, **kwargs))
        new_._grad_op = ReshapeGrad(self._array)
        return new_
    
    @property
    def shape(self) -> Tuple[int]:
        if self._array is None:
            raise ValueError("initialization is required")
        return self._array.shape
    
    @property
    def n_dim(self) -> int: 
        return len(self.shape)
    
    @property
    def T(self) -> 'GradArray':
        grad_T = None if self._grad is None else self._grad.T
        return GradArray(self._array.copy().T, grad_T, grad_op=TransposeGrad(self), name=f'{self._name}.T')
    
    def item(self) -> Number: 
        return self._array.item()
    
    def grad_item(self) -> Number:
        return self._grad.item()

    def __add__(self, rhs: Union[Number, 'GradArray']) -> 'GradArray':
        if isinstance(rhs, Number): 
            rhs = GradArray(np.array(rhs, dtype=np.float64))
        return GradArray(self._array + rhs._array, grad_op=AddGrad(self, rhs), name=f'{self._name} + {rhs._name}')
    
    def __radd__(self, lhs: Union[Number, 'GradArray']) -> 'GradArray':
        if isinstance(lhs, Number):
            lhs = GradArray(np.array(lhs, dtype=np.float64))
        return GradArray(lhs._array + self._array, grad_op=AddGrad(lhs, self), name=f'{lhs._name} + {self._name}')
    
    def __sub__(self, rhs: Union[Number, 'GradArray']) -> 'GradArray':
        if isinstance(rhs, Number):
            rhs = GradArray(np.array(rhs, dtype=np.float64))
        return GradArray(self._array - rhs._array, grad_op=AddGrad(self, -rhs), name=f'{self._name} - {rhs._name}')
    
    def __rsub__(self, lhs: Union[Number, 'GradArray']) -> 'GradArray':
        if isinstance(lhs, Number):
            lhs = GradArray(np.array(lhs, dtype=np.float64))
        return GradArray(lhs._array - self._array, grad_op=AddGrad(lhs, -self), name=f'{lhs._name} - {self._name}')
    
    def __mul__(self, rhs: Union[Number, 'GradArray']) -> 'GradArray':
        if isinstance(rhs, Number):
            rhs = GradArray(np.array(rhs, dtype=np.float64))
            grad = ScalarMulGrad(rhs, self)
        else: # FIXME: scalar multiplication often becomes element-wise multiplication
            grad = ElemMulGrad(self, rhs)
        return GradArray(self._array * rhs._array, grad_op=grad, name=f'{self._name} * {rhs._name}')
    
    def __rmul__(self, lhs: Union[Number, 'GradArray']) -> 'GradArray':
        if isinstance(lhs, Number):
            lhs = GradArray(np.array(lhs, dtype=np.float64))
            grad = ScalarMulGrad(lhs, self)
        else: 
            grad = ElemMulGrad(lhs, self)
        return GradArray(lhs._array * self._array, grad_op=grad, name=f'{lhs._name} * {self._name}')
    
    def __truediv__(self, rhs: Union[Number, 'GradArray']) -> 'GradArray':
        if isinstance(rhs, Number): 
            rhs = GradArray(np.array(rhs, dtype=np.float64))
        return self * recip(rhs)
    
    def __rtruediv__(self, lhs: Union[Number, 'GradArray']) -> 'GradArray':
        if isinstance(lhs, Number):
            lhs = GradArray(np.array(lhs, dtype=np.float64))
        inv = recip(self)
        return lhs * inv
    
    def __matmul__(self, rhs: 'GradArray') -> 'GradArray':
        return GradArray(self._array @ rhs._array, grad_op=MatMulGrad(self, rhs), name=f'{self._name} @ {rhs._name}')
    
    def __rmatmul__(self, lhs: 'GradArray') -> 'GradArray':
        return GradArray(lhs._array @ self._array, grad_op=MatMulGrad(lhs, self), name=f'{lhs._name} @ {self._name}')
    
    def __neg__(self) -> 'GradArray':
        minus = GradArray(np.array(-1, dtype=np.float64))
        return GradArray(-self._array.copy(), grad_op=ScalarMulGrad(minus, self), name=f'-{self._name}')
    
    def __pow__(self, exponent: float) -> 'GradArray':
        return GradArray(np.power(self._array, exponent), grad_op=PowerGrad(self, exponent), name=f'{self._name} ** {exponent}')
    
    def __repr__(self) -> str:
        return f'GradArray(name={self._name}, shape={self.shape}, grad_op={self._grad_op.__class__.__name__})'

def expand(arr: GradArray, dim: int) -> GradArray: # only supports vector to 2-dim array (matrix)
    if arr.n_dim > 1: 
        raise ValueError("expand only works for 1-dim array (vector)")
    return GradArray(np.tile(arr._array, reps=(dim, 1)), grad_op=ExpandGrad(arr))

def sum(arr: GradArray, axis: int) -> GradArray: 
    return GradArray(np.sum(arr._array, axis=axis), grad_op=SumGrad(arr))

# TODO: y GradArray support
# TODO: axis support
def min(arr: GradArray, y: float) -> GradArray: 
    return GradArray(np.minimum(arr._array, y), grad_op=MinGrad(arr, y))

def max(arr: GradArray, y: float) -> GradArray: 
    return GradArray(np.maximum(arr._array, y), grad_op=MaxGrad(arr, y))

def l2_norm_square(arr: GradArray, axis: int) -> GradArray: 
    return sum(arr ** 2, axis=axis)

def exp(arr: GradArray) -> GradArray: 
    return GradArray(np.exp(arr._array), grad_op=ExpGrad(arr))

def recip(arr: GradArray) -> GradArray: 
    return GradArray(np.reciprocal(arr._array), grad_op=RecipGrad(arr))
