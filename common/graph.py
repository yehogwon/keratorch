import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tempfile
import graphviz

from common.array import GradArray

class GraphNode: 
    def __init__(self, array: GradArray=None) -> None:
        self.array = array
        self.children = []

    def add_child(self, child: 'GraphNode') -> None: 
        self.children.append(child)
    
    def __repr__(self) -> str:
        return f'GraphNode({self.array} : ({len(self.children)})'
    
    def backward_string(self, prefix: str='') -> str: 
        string = prefix + str(self) + '\n'
        for child in self.children: 
            string += child.deep_string(prefix + '-- ')
        return string
    
    def __dot_language(self) -> str: 
        dot = ''
        dot += f'"{id(self.array)}" [ label = "{self.array._name} {self.array.shape}" ]'
        if self.array._grad_op is not None:
            dot += f'"{id(self.array._grad_op)}" [ label = "{self.array._grad_op}", fillcolor = "gray", style=filled ]'
        for child in self.children: 
            dot += f'{id(self.array)} -> {id(self.array._grad_op)}; \n'
            dot += f'{id(self.array._grad_op)} -> {id(child.array)}; \n'
            dot += child.__dot_language()
        return dot

    def dot_language(self) -> str: 
        string = 'digraph G {\n'
        string += self.__dot_language()
        string += '}'
        return string
    
    def dot_graph(self) -> graphviz.Digraph: 
        return graphviz.Source(self.dot_language())
    
    def dot_graph_show(self) -> None: 
        graph_ = self.dot_graph()
        with tempfile.NamedTemporaryFile(suffix='.gv') as f:
            print(f.name)
            f.write(graph_.source.encode('utf-8'))
            f.flush()
            graph_.render(f.name, view=True)

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
