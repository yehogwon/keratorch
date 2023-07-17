import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from common.array import GradArray
from common.layer import Layer, Linear
from common.loss import MSE
from common.optimizer import SGD
from common.activation import Sigmoid

import matplotlib.pyplot as plt
from tqdm import tqdm

class BinaryClassifier(Layer): 
    def __init__(self) -> None:
        self.fc = Linear(2, 1)
        self.sigmoid = Sigmoid()
        
    def forward(self, x: GradArray) -> GradArray: 
        return (self.sigmoid(self.fc(x)))

network = BinaryClassifier()
loss = MSE()
optim = SGD(network.get_params(), lr=0.05)

x = GradArray(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
gt = GradArray(np.array([[0], [1], [1], [1]])) # OR operation

loss_list = []
for i in tqdm(range(5000)):
    y = network(x)
    loss(y, gt)
    loss_list.append((i, loss.out.item()))
    loss.backward()
    optim.step()

print(network(x)._array)
print(network(x)._array.round())

plt.plot(*zip(*loss_list))
plt.show()
