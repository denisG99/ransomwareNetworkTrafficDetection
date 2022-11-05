from dataclasses import dataclass, field

from NeuralNetwork import NeuralNetwork
from NodeType import NodeType
from Classification import Classification

from sklearn.model_selection import train_test_split

#from numpy.random import MT19937 #Mersenne Twister pseudo-random number generator
#from numpy.random import RandomState
import numpy as np

import pandas as pd

@dataclass
class Node:
    #fields
    __nn : NeuralNetwork = field(init=False) #perceptron definition
    __patterns: np.ndarray #field containing training data for perceptron
    __num_features: int #number of features into dataset
    __label: Classification = Classification.NONE
    __type : NodeType = NodeType.DECISION
    __child : list = field(init=False, default_factory=list) #list containing childs nodes
    __depth : int = 0 #TODO: forse da spostare
    __threshold : float = 0.0
    __toler : float = 0.0 #tollerance that indicate the end of the branch training. Will be use in convergence test #TODO:forse da spostare
    __wait_epochs : int = 0 #wait epochs that wait before splitting node if the boundaries don't get any improvement #TODO:forse da spostare
    __num_classi : int = 2

    def __post_init__(self):
        self.__nn = NeuralNetwork(self.__num_features)

    def get_num_features(self) -> int:
        return self.__num_features

    #TODO: da finire
    def train(self, epochs: int = 1) -> None:
        data = self.__patterns[:, 0 : self.__num_features]  # patterns dataset
        target = self.__patterns[:, self.__num_features]  # labels dataset
        #TODO: rendere suddivisione riproducibile
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.3,train_size=.7)  # dataset split with 70-30 splitting rule

        self.__nn.fit(X_train, y_train, epochs)

        #TODO: splitting dataset based on prediction to create LTS


        print(f'Train accuracy:')
        self.__nn.evaluate(X_train, y_train)

        print(f'Test accuracy:')
        self.__nn.evaluate(X_test, y_test)

    def split_node(self) -> None:
        self.__type = NodeType.SPLIT
        left_patterns, right_patterns = np.zeros(1), np.zeros(1) #TODO: da fare funzione che dividi a metà il dataset
        left_node, right_node = Node(left_patterns, self.__num_features), Node(right_patterns, self.__num_features)

        #TODO: controlla l'omogeneità dell' LTS per classificare la tipologia di nodo
        if None:
            pass

        self.__child.append(left_node)
        self.__child.append(right_node)

    def predict(self, X: np.ndarray) -> int:
        pass


#-----------------------------------------------------------------------------------------------------------------------

def main() -> None:
    patterns = pd.read_csv("toydata.csv").to_numpy()
    root = Node(patterns, patterns.shape[1] - 1)

    root.train(5)

if __name__ == "__main__":
    main()