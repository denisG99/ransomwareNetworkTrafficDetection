import os
import sys
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.abspath('./model'))

from model.NeuralTree import NeuralTree
from model.NodeType import NodeType
from model.Node import Node


def __nodes_edges(root: Node, nodes: np.ndarray, edges: list):
    node_id = root.get_id

    nodes = np.append(nodes, [node_id])

    if not root.get_type() == NodeType.LEAF:
        left, right = root.get_left(), root.get_right()

        edges.append([(node_id, left.get_id()), (node_id, right.get_id())])

        nodes, edges = __nodes_edges(left, nodes, edges)
        nodes, edges = __nodes_edges(right, nodes, edges)

    return nodes, edges

def main():
    nt = NeuralTree.load_model("../exported_model/nt_unbalance10.pkl")

    tree = nx.Graph()
    nodes, edges = __nodes_edges(nt.get_root(), np.array([]), list([]))

    print(nodes)
    print(edges)

    tree.add_nodes_from(nodes)
    tree.add_edges_from(edges)



if __name__ == "__main__":
    main()


