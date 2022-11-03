from dataclasses import dataclass, field
from Perceptron import Perceptron
from NodeType import NodeType
from Classification import Classification
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

@dataclass
class Node:
    #fields
    __perceptron : Perceptron = field(init=False) #perceptron definition
    __patterns: np.ndarray #field containing training data for perceptron
    __num_features: int #number of features into dataset
    __label: Classification = Classification.NONE
    __type : NodeType = NodeType.DECISION
    __child : list = field(init=False, default_factory=list) #list containing childs nodes
    __depth : int = 0 #TODO: forse da togliere
    __threshold: float = 0.0
    __toler: float = 0.0 #tollerance that indicate the end of the branch trainig. Will be use in convergence test #TODO:forse da togliere
    __wait_epochs: int = 0 #wait epochs that wait before splitting node if the boundaries don't get any improvement #TODO:forse da togliere

    def __post_init__(self):
        self.__perceptron = Perceptron(self.__num_features)

    def train(self, epochs: int = 1) -> None:
        data = self.__patterns[:, 0 : self.__num_features] #patterns dataset
        target = self.__patterns[:, self.__num_features] #labels dataset

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.3, train_size=.7) #dataset split with 70-30 splitting rule
        self.__perceptron.fit(X_train, y_train, epochs)

        #print(f'Train accuracy: {self.__perceptron.evaluate(X_train, y_train) * 100}')
        #print(f'Test accuracy: {self.__perceptron.evaluate(X_test, y_test) * 100}')

#-----------------------------------------------------------------------------------------------------------------------

def main() -> None:
    patterns = pd.read_csv("toydata.csv").to_numpy()
    root = Node(patterns, patterns.shape[1] - 1)

    root.train()


if __name__ == "__main__":
    main()