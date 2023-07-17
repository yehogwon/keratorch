import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from common.array import GradArray

def he_initialize(x: GradArray) -> None:
    x._array = np.random.normal(0, 2 / x.shape[1], x.shape).astype(x._array.dtype)

def zero_initialize(x: GradArray) -> None:
    x._array = np.zeros(x.shape, dtype=x._array.dtype)
