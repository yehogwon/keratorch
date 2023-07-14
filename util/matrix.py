import numpy as np

def rot180(x: np.ndarray) -> np.ndarray:
    return np.rot90(x, k=2, axes=(x.ndim - 2, x.ndim - 1))

def convolve(x: np.ndarray, filter: np.ndarray, stride: int, padding: int) -> np.ndarray: 
    return cross_correlate(x, np.flip(filter), stride, padding)

def cross_correlate(x: np.ndarray, filter: np.ndarray, stride: int, padding: int) -> np.ndarray: 
    if filter.ndim == 2 and filter.shape[0] != filter.shape[1] or filter.ndim == 3 and filter.shape[1] != filter.shape[2] or filter.ndim == 1 or filter.ndim > 3: 
        raise ValueError("Filter must be 2d/3d-square matrix")
    raise NotImplementedError
