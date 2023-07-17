import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
from random import randint

from common.array import GradArray

class GraphNode: 
    def __init__(self, array: GradArray=None) -> None:
        self.array = array
        self.children = []

    def add_child(self, child: 'GraphNode') -> None: 
        self.children.append(child)
    
    def __repr__(self) -> str:
        return f'GraphNode({self.array} : ({len(self.children)})'
    
    def deep_string(self, prefix: str='') -> str: 
        string = prefix + str(self) + '\n'
        for child in self.children: 
            string += child.deep_string(prefix + '-- ')
        return string

def backward_graph(arr: GradArray) -> GraphNode:
    root = GraphNode(arr)
    leaves = [root]
    while leaves: 
        leaf = leaves.pop()
        grad_op = leaf.array._grad_op
        if grad_op is None:
            continue
        for in_arr in grad_op._inputs: 
            node = GraphNode(in_arr)
            leaf.add_child(node)
            leaves.append(node)
    return root
