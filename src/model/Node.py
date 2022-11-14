from dataclasses import dataclass, field
from typing import Tuple, Union, Dict, Any, Iterable
from sklearn.model_selection import train_test_split
from scipy.stats import entropy

from NeuralNetwork import NeuralNetwork
from NodeType import NodeType
from Classification import Classification

import numpy as np

ENTROPY_TH: float = 0.1

@dataclass
class Node:
    #fields
    __nn : NeuralNetwork = field(init=False) #perceptron definition
    __patterns: np.ndarray #field containing training data for perceptron
    __num_features: int #number of features into dataset
    __entropy: float = field(init=False)  # dataset entropy
    __label: Classification = Classification.NONE
    __type : NodeType = NodeType.DECISION
    __childs : np.ndarray= field(init=False) #list containing childs nodes
    #__depth : int = 0 #TODO: forse da spostare
    __threshold : float = 0.0
    #__toler : float = 0.0 #tollerance that indicate the end of the branch training. Will be use in convergence test #TODO:forse da spostare
    #__wait_epochs : int = 0 #wait epochs that wait before splitting node if the boundaries don't get any improvement #TODO:forse da spostare
    __num_classi : int = 2

    def __post_init__(self):
        self.__nn = NeuralNetwork(self.__num_features)
        self.__entropy, occurs = self.__compute_entropy()
        self.__childs = np.array([], dtype=Node)

        if self.__entropy <= ENTROPY_TH:
            self.__type = NodeType.LEAF
            self.__label = Classification(max(occurs, key=occurs.get))

    def get_num_features(self) -> int:
        return self.__num_features

    def __compute_entropy(self) -> Tuple[Union[float, np.ndarray, Iterable, int], Dict[Any, Any]]:
        probs = list()
        label = self.__patterns[:, self.__num_features]
        labels, counts = np.unique(label, return_counts=True)

        for count in counts:
            probs.append(count / len(label))

        #label = dict(zip(label, count))
        #print(probs)

        return entropy(probs, base=2), dict(zip(labels, counts))

    #TODO: da finire
    def train(self) -> None:
        if not self.__type == NodeType.LEAF:
            data = self.__patterns[:, 0 : self.__num_features]  # patterns dataset
            target = self.__patterns[:, self.__num_features]  # labels dataset
            #TODO: rendere suddivisione riproducibile
            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.3,train_size=.7)  # dataset split with 70-30 splitting rule

            self.__nn.fit(X_train, y_train) #TODO: ipermarametrizzazione
            #self.__nn.evaluate(X_train, y_train, "Train accuracy")
            #self.__nn.evaluate(X_test, y_test, "Test accuracy")

            goodware_lts, malware_lts = self.__dataset_split(self.__patterns, data)

            #TODO: realizzare creazione nodo tenendo conto delle caratteristiche degli LTS
            self.__childs = np.append(self.__childs, [Node(malware_lts, self.__num_features), Node(goodware_lts, self.__num_features)])

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
        lts0 = np.array([])
        lts1 = np.array([])

        for pattern, row in zip(X, lts):
            #print(pattern)
            #print(row)
            if self.predict(pattern) == 1.0:
                lts1 = np.append(lts1, row)
            else:
                lts0 = np.append(lts0, row)

        lts0 = np.resize(lts0, (int(len(lts0) / (self.__num_features + 1)), self.__num_features + 1))
        lts1 = np.resize(lts1, (int(len(lts1) / (self.__num_features + 1)), self.__num_features + 1))

        #print(lts0)
        #print(lts1)

        #row0, row1 = lts0.shape[0], lts1.shape[0]

        #return lts0.reshape(row0, self.__num_features + 1), lts1.reshape(row1, self.__num_features + 1)
        return lts0, lts1

    def predict(self, X) -> float:
        return self.__nn.predict(X.reshape(1, -1))


#-----------------------------------------------------------------------------------------------------------------------

def main() -> None:
    import pandas as pd
    import matplotlib.pyplot as plt


    patterns = pd.read_csv("toydata.csv", header=None).to_numpy()
    print(patterns.shape)

    root = Node(patterns, patterns.shape[1] - 1)
    #X, y = patterns[:, :2], patterns[:, 2]

    root.train()

    print(root)
    """
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

    lts0, lts1 = root.dataset_split(patterns, X)
    #root.dataset_split(patterns, X)

    X0, y0, X1, y1 = lts0[:, :2], lts0[:, 2], lts1[:, :2], lts1[:, 2]

    plt.scatter(X0[y0 == 0, 0], X0[y0 == 0, 1], label='class 0', marker='o')
    plt.scatter(X0[y0 == 1, 0], X0[y0 == 1, 1], label='class 1', marker='s')
    plt.title('LTS_0')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.legend()
    plt.show()

    plt.scatter(X1[y1 == 0, 0], X1[y1 == 0, 1], label='class 0', marker='o')
    plt.scatter(X1[y1 == 1, 0], X1[y1 == 1, 1], label='class 1', marker='s')
    plt.title('LTS_1')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.legend()
    plt.show()
    """

if __name__ == "__main__":
    main()