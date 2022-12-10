from Node import Node
from Classification import Classification

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np
import pickle as pkl

@dataclass
class NeuralTree:
    """
    Costruttore:
        NeuralTree(__data_path,
                   __wait_epochs)
        """
    __root : Node = field(init=False)
    __data_path : str
    #TREE STATISTICS
    """
    __depth: int = 0 #depth of tree
    __num_nodes : int = 0 #count the number of node
    __num_splitted : int = 0 #count the number of splittin node
    __num_leaf : int = 0 #count the number of leaf node
    """
    #__toler : float = 0.0 #tollerance that indicate the end of the branch training. Will be use in convergence test
    __wait_epochs : int = 0 #wait epochs that wait before splitting node if the boundaries don't get any improvement

    def __post_init__(self):
        data = pd.read_csv(self.__data_path, low_memory=False).to_numpy()

        self.__root = Node(data, data.shape[1] - 1)
        #self.__num_nodes += 1

    def train(self, epochs: int, verbose: int = 0) -> None:
            self.__root.train(epochs, self.__wait_epochs, verbose=verbose)

    def make_prediction(self, sample: np.ndarray, verbose: int = 0) -> Classification:
        return self.__root.predict(sample.reshape((1, -1)), verbose=verbose)

    def save_model(self) -> None:
        """
        This function save the model in pickel file format

        :return: a pickel file contains the neural tree called 'model.pkl'
        """
        with open('model.pkl', 'wb') as file:
            pkl.dump(self, file)

        print("SALVATAGGIO COMPLETATO!")

    def load_model(self, pkl_path: str) -> Any:
        """
        This function reload a model by pickel file

        :param pkl_path: string containing pickle file path

        :return: neural tree data structure that has been reload by pickle file
        """
        with open(pkl_path, 'rb') as file:
            nt = pkl.load(file)

        print("CARICAMENTO COMPLETATO!")

        return nt

#-----------------------------------------------------------------------------------------------------------------------

import time

def main() -> None:
    start_time = time.time()
    nt = NeuralTree("./toydata.csv", 5)

    nt.train(250, verbose=0)

    print(f"Time for training: {time.time() - start_time} secondi")
    print(nt.make_prediction(np.array([0, 0]), verbose=0))

if __name__ == "__main__":
    main()