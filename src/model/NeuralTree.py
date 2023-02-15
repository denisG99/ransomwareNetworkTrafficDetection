from numpy import ndarray

from Node import Node
from Classification import Classification

from dataclasses import dataclass, field
from typing import Any, Tuple

import pandas as pd
import numpy as np
import pickle as pkl
#import cv2 as cv

@dataclass
class NeuralTree:
    """
    Costruttore:
        NeuralTree(__entropy,
                   __num_features,
                   __cardinality,
                   __wait_epochs = 0)
        """
    __root : Node = field(init=False)
    __entropy: float = field(repr=False)
    __num_features : int
    __cardinality : int
    #__data_path : str = field(repr=False)
    #TREE STATISTICS
    """
    __depth: int = 0 #depth of tree
    __num_nodes : int = 0 #count the number of node
    __num_splitted : int = 0 #count the number of splittin node
    __num_leaf : int = 0 #count the number of leaf node
    """
    #__toler : float = 0.0 #tollerance that indicate the end of the branch training. Will be use in convergence test
    __wait_epochs : int = 0 #wait epochs that wait before splitting node if the boundaries don't get any improvement

    def get_root(self):
        return self.__root

    def __post_init__(self):
        #data = pd.read_csv(self.__data_path, low_memory=False).to_numpy()

        #self.__root = Node(data.shape[1] - 1)
        self.__root = Node(self.__entropy, self.__num_features, self.__cardinality)
        del self.__entropy
        #self.__num_nodes += 1

    def train(self,data: np.ndarray, epochs: int, verbose: int = 0) -> None:
        self.__root.train(data, epochs, self.__wait_epochs, verbose=verbose)

    def make_predictions(self, samples: np.ndarray, verbose: int = 0):
        #preds = np.array([])
        #tot_time = 0

        #for sample in samples:
         #   pred = self.__root.predict(sample.reshape(1, -1), verbose=verbose)
          #  preds = np.append(preds, pred)
            #tot_time += time

        #return preds#, tot_time
        return np.apply_along_axis(self.__root.predict, 1, samples)


    #TODO: ampliare le metriche di valutazione nel momento in cui serva
    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: int = 0) -> float:
        predictions: np.ndarray = np.array([])

        #TODO: trovare modo per togliere for sfruttando funzioni offerte da numpy
        for record in X:
            predictions = np.append(predictions, [self.make_predictions(record, verbose=verbose).value])

        return (np.sum(y==predictions) / predictions.shape[0]) * 100

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

#-----------------------------------------------------------------------------------------------------------------------
"""
import time

def main() -> None:
    start_time = time.time()
    nt = NeuralTree("./toydata.csv", 5)

    nt.train(250, verbose=0)

    print(f"Time for training: {time.time() - start_time} secondi")
    #print(nt.make_predictions(np.array([0, 0]), verbose=0))

    #VALUTAZIONE MODELLO
    data = pd.read_csv("./toydata.csv", header=None).to_numpy()

    print(f"Precision: {nt.evaluate(data[:, 0 : 2], data[:, 2], verbose=0)}%")
    #nt.visualize(1080, 1920)

if __name__ == "__main__":
    main()
"""