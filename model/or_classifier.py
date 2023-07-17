import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from common.array import GradArray
from common.layer import Layer, Linear
from common.loss import MSE
from common.optimizer import SGD
from common.activation import Sigmoid, Tanh
from common.graph import backward_graph

import matplotlib.pyplot as plt
from tqdm import tqdm

class BinaryClassifier(Layer): 
    def __init__(self) -> None:
        self.fc = Linear(2, 1)
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        
    def forward(self, x: GradArray) -> GradArray: 
        return self.sigmoid(self.fc(x))
        # return (self.tanh(self.fc(x)) + 1) / 2

network = BinaryClassifier()
loss = MSE()
optim = SGD(network.get_params(), lr=0.05)

x = GradArray(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), name='x')
gt = GradArray(np.array([[0], [1], [1], [1]]), name='gt') # OR operation

backward_graph(network(x)).dot_graph().view()

loss_list = []
for i in tqdm(range(1000)):
    y = network(x)
    loss(y, gt)
    loss_list.append((i, loss.out.item()))
    loss.backward()
    optim.step()

out = network(x)
print(out, out.shape)
print(out._array.round())

plt.plot(*zip(*loss_list))
plt.show()
