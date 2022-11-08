from dataclasses import dataclass, field
from typing import Tuple

from NeuralNetwork import NeuralNetwork
from NodeType import NodeType
from Classification import Classification

from sklearn.model_selection import train_test_split

#from numpy.random import MT19937 #Mersenne Twister pseudo-random number generator
#from numpy.random import RandomState
import numpy as np

@dataclass
class Node:
    #fields
    __nn : NeuralNetwork = field(init=False) #perceptron definition
    __patterns: np.ndarray #field containing training data for perceptron
    __num_features: int #number of features into dataset
    __label: Classification = Classification.NONE
    __type : NodeType = NodeType.DECISION
    __childs : list = field(init=False, default_factory=list) #list containing childs nodes
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
    def train(self) -> None:
        data = self.__patterns[:, 0 : self.__num_features]  # patterns dataset
        target = self.__patterns[:, self.__num_features]  # labels dataset
        #TODO: rendere suddivisione riproducibile
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.3,train_size=.7)  # dataset split with 70-30 splitting rule

        self.__nn.fit(X_train, y_train)
        self.__nn.evaluate(X_train, y_train, "Train accuracy")
        self.__nn.evaluate(X_test, y_test, "Test accuracy")

        goodware_lts, malware_lts = self.__dataset_split(self.__patterns, data)

        #TODO: realizzare creazione nodo tenendo conto delle caratteristiche degli LTS


        np.append(self.__childs, [Node(malware_lts, self.__num_features), Node(goodware_lts, self.__num_features)])

        for child in self.__childs:
            if not self.__type == NodeType.LEAF:
                child.train()

    """"
    def split_node(self) -> None:
        self.__type = NodeType.SPLIT
        left_patterns, right_patterns = np.zeros(1), np.zeros(1) #TODO: da fare funzione che dividi a metà il dataset
        left_node, right_node = Node(left_patterns, self.__num_features), Node(right_patterns, self.__num_features)

        #TODO: controlla l'omogeneità dell' LTS per classificare la tipologia di nodo
        if None:
            pass

        self.__child.append(left_node)
        self.__child.append(right_node)
    """

    def __dataset_split(self, lts: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function create a local datasets starting from starting dataset stored in current node based on decision boundary

        IDEA:
          foreach patterns into dataset will be compute the prediction and based on that the dataset will split

        --------------

        :param lts: Local Training Set that will be split
        :param X: patterns whitch make predictions

        :return: return an array containig the two dataset split
        """
        lts0 = np.array({}, dtype=np.ndarray)
        lts1 = np.array({}, dtype=np.ndarray)

        for pattern, row in zip(X, lts):
            if self.__nn.predict(pattern) == 1.0:
                np.append(lts1, row)
            else:
                np.append(lts0, row)

        return lts0, lts1

    def predict(self, X) -> float:
        return self.__nn.predict(X)


#-----------------------------------------------------------------------------------------------------------------------

def main() -> None:
    import pandas as pd
    import matplotlib.pyplot as plt


    patterns = pd.read_csv("toydata.csv").to_numpy()
    root = Node(patterns, patterns.shape[1] - 1)
    X, y = patterns[:, :2], patterns[:, 2]

    #plotting
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='class 0', marker='o')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='class 1', marker='s')
    plt.title('Dataset')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.legend()
    plt.show()

    root.train()

if __name__ == "__main__":
    main()