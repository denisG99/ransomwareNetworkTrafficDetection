from Node import Node

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pickle as pkl

@dataclass
class NeuralTree:
    """
    Costruttore:
        NeuralTree(__entropy,
                   __num_features,
                   __cardinality,
                   __wait_epochs = 0,
                   __depth = 0,
                   __num_nodes = 0,
                   __num_splitted = 0,
                   __num_substitution = 0,
                   __num_leaf = 0)
        """
    __root : Node = field(init=False)
    __entropy: float = field(repr=False)
    __num_features : int
    __cardinality : int
    #TREE STATISTICS
    __depth: int = 0 #depth of tree
    __num_nodes : int = 0 #count the number of node
    __num_splitted : int = 0 #count the number of splittin node
    __num_substitution : int = 0 #count the number of perceptron substitution
    __num_leaf : int = 0 #count the number of leaf node
    __wait_epochs : int = 0 #wait epochs that wait before splitting node if the boundaries don't get any improvement

    def get_root(self):
        return self.__root

    def __post_init__(self):
        self.__root = Node(self.__entropy, self.__num_features, self.__cardinality)
        del self.__entropy
        self.__num_nodes += 1

    def train(self,data: np.ndarray, epochs: int, verbose: int = 0) -> None:
        depth, num_nodes, num_splitted, num_substitution, num_leaf = self.__root.train(data, epochs, self.__wait_epochs, verbose,
                                                                                       self.__depth, self.__num_nodes, self.__num_splitted,
                                                                                       self.__num_substitution, self.__num_leaf)

        self.__depth = depth
        self.__num_nodes = num_nodes
        self.__num_splitted = num_splitted
        self.__num_substitution = num_substitution
        self.__num_leaf = num_leaf

    def make_predictions(self, samples: np.ndarray):
        return np.apply_along_axis(self.__root.predict, 1, samples)

    def save_model(self, path: str) -> None:
        """
        This function save the model in pickel file format

        :return: a pickel file contains the neural tree called 'model.pkl'
        """
        with open(path, 'wb') as file:
            pkl.dump(self, file)

        print("SALVATAGGIO COMPLETATO!")

    @staticmethod
    def load_model(pkl_path: str) -> Any:
        """
        This function reload a model by pickel file

        :param pkl_path: string containing pickle file path

        :return: neural tree data structure that has been reload by pickle file
        """
        with open(pkl_path, 'rb') as file:
            nt = pkl.load(file)

        print("CARICAMENTO COMPLETATO!")

        return nt

    def tree_statistic(self):
        print("STATISTICHE DEL NEURAL TREE")
        print(f"profondità: {self.__depth}")
        print(f"numero totale dei nodi: {self.__num_nodes}")
        print(f"numero di nodi di split: {self.__num_splitted}")
        print(f"perceptron substitution: {self.__num_substitution}")
        print(f"numero nodi foglia: {self.__num_leaf}")