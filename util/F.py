import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray: 
    return 1 / (1 + np.exp(-x))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def relu(x: np.ndarray) -> np.ndarray: 
    return np.maximum(0, x)

def l2_norm(x: np.ndarray) -> np.ndarray: 
    return np.linalg.norm(x)
