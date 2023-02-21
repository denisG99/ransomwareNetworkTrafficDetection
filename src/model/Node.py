from dataclasses import dataclass, field
from typing import Tuple, Union, Dict, Any, Iterable

from sklearn.metrics import confusion_matrix

from scipy.stats import entropy

from NeuralNetwork import NeuralNetwork
from NodeType import NodeType
from Classification import Classification
import Node


import numpy as np
import uuid
import math

@dataclass
class Node:
    """
    Costruttore:
        Node(__entropy,
             __num_features,
             __cardinality,
             __label = Classification.NONE,
             __level = 0)
    """
    __id: str = field(init=False, repr=False) #node identifier used in logging
    __nn : NeuralNetwork = field(init=False, repr=False) #perceptron definition
    __left: Node = field(init=False) #left child
    __right: Node = field(init=False) #rigth child
    __is_splitted: bool = field(init=False)  #indicate if node is split
    __type: NodeType = field(default=NodeType.DECISION, init=False)
    __entropy: float # dataset entropy
    __num_features: int  #number of features into dataset
    __cardinality: int # dataset cardinality
    __label: Classification = Classification.NONE
    __level : int = 0

    # GETTER & SETTER
    def get_num_features(self) -> int:
        return self.__num_features

    def get_left(self):
        return self.__left

    def get_right(self):
        return self.__right

    def get_type(self) -> NodeType:
        return self.__type

    def get_label(self):
        return self.__label

    def get_id(self):
        return self.__id

#-----------------------------------------------------------------------

    def __post_init__(self):
        self.__id = str(uuid.uuid4())
        self.__nn = NeuralNetwork(self.__num_features)
        self.__left = None
        self.__right = None
        self.__is_splitted = False

        if self.__is_homogeneous(self.__cardinality):
            self.__type = NodeType.LEAF
            self.__num_classi = 1
            self.__nn.set_num_classi(1)
        else:
            self.__label = Classification.NONE

    def __compute_entropy(self, data) -> Tuple[Union[float, np.ndarray, Iterable, int], Dict[Any, Any]]:
        probs = list()
        label = data[:, self.__num_features]
        labels, counts = np.unique(label, return_counts=True)

        for count in counts:
            probs.append(count / len(label))

        return entropy(probs, base=2), dict(zip(labels, counts))

    def __is_homogeneous(self, n) -> bool:
        """
        This function check if the dataset may be considered homogeneous. To do this you need to define some kind of
        threshold which depends on dataset cardinality.

        The threshold is thus defined:
                entropy_th = log2(n)/n, dove n è la cardinalità del dataset

        :return: true if the dataset is homogeneous, otherwise false
        """
        return self.__entropy <= (math.log2(n) / n)

    def train(self, data: np.ndarray, depth: int = 0, num_nodes: int = 0, num_splitted: int = 0, num_substitution: int = 0, num_leaf: int = 0):
        if not self.__type == NodeType.LEAF:
            X = data[:, 0 : self.__num_features]  # patterns dataset
            y = data[:, self.__num_features]  # labels dataset

            self.__nn.fit(X, y)

            #DATASET SPLITTING
            preds = self.__nn.predict(X)
            print(f"Bias dopo addestrameto: {self.__nn.get_weight()[1]}")

            if not len(np.unique(preds)) == 1:
                self.__make_acceptable_model(X, y, preds) #create trained perceptron that considered acceptable (perceptron substitution)
                print(f"Bias accettabile: {self.__nn.get_weight()[1]}")

                if self.__type == NodeType.SUBSTITUTION:
                    num_substitution += 1

                lts0, lts1 = self.__dataset_split(data, X)
            else: #split based on split node
                print("Creazione split rule")
                lts0, lts1 = self.__create_split_node(data, X)
                num_splitted += 1

            del data, X, y, preds #FIX for momory issues

            #CHILD CREATION + TRAINING
            entropy, occurs = self.__compute_entropy(lts1)
            self.__left = Node(entropy, self.__num_features, lts1.shape[0], Classification(max(occurs, key=occurs.get)), self.__level + 1)

            entropy, occurs = self.__compute_entropy(lts0)
            self.__right = Node(entropy, self.__num_features, lts0.shape[0], Classification(max(occurs, key=occurs.get)), self.__level + 1)
            depth = max(depth, self.__level + 1)
            num_nodes += 2

            depth, num_nodes, num_splitted, num_substitution, num_leaf = self.__left.train(lts1, num_nodes, num_splitted, num_substitution, num_leaf)
            depth, num_nodes, num_splitted, num_substitution, num_leaf = self.__right.train(lts0, depth, num_nodes, num_splitted, num_substitution, num_leaf)

            print(f"END TRAINING Node {self.__id}")
        else:
            num_leaf += 1
        print(f"Node {self.__id} is {self.__type} Node (Label -> {self.__label})")

        return depth, num_nodes, num_splitted, num_substitution, num_leaf

    #TODO: da sistemare
    def __make_acceptable_model(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        This function create model created by perceptron that is considered acceptable. To do this I have to check the follow condition:
            E_t <= E_0 / 2 and (E_max - E_min) <= E_t

            E_t = 1 - Kc/kt, where Kc is number of correctly classified pattern and Kt is the total number of pattern into node
            E_0 -> error of the trained perceptron
            E_max = max{E_i}
            E_min = min{E_i}
            E_i = 1 - K_ci / K_ti, where K_ci is number of correctly classified pattern of class i and K_ti is total number of pattern classified as class i

        For calculate that value we will use a confusion matrix(cm) in fact:
            * Kc -> sum of the diagonal
            * K_ci -> cm[i][i]
            * K_ti -> sum of i-th row

        :param y_true: array containing true label
        :param y_pred: array containing predicted label

        :return: true if the two partitions are almost balanced, otherwise
        """
        old_weight = self.__nn.get_weight()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        K_t = y_true.shape[0]

        K_c = tn + tp
        E_0 = 1 - (K_c / K_t)

        # compute centroid of the local training set
        centroid = self.__get_centroid(X)


        # compute the new hyperplane passing throw centroid
        hyperplane = np.append(self.__nn.get_weight()[: self.__num_features], centroid.dot(self.__nn.get_weight()[: self.__num_features]))
        self.__nn.reinit_weights(hyperplane) # perceptron substitution

        preds = self.__nn.predict(X)

        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

        K_c = tn + tp
        E_t = 1 - (K_c / K_t)

        K_ci = np.array([tp, tn])
        K_ti = np.array([tp + fp, tn + fn])

        E_i = 1 - np.divide(K_ci, K_ti)

        E_max, E_min = np.max(E_i), np.min(E_i)
        print(f"E_0 = {E_0}")
        print(f"E_0 / 2 = {E_0/2}")
        print(f"E_t = {E_t}")
        print(f"K_ci = {K_ci}")
        print(f"K_ti = {K_ti}")
        print(f"E_max - E_min = {E_max - E_min}")


        #TODO: ricontrollare condizione
        if E_t <= (E_0 / 2) and (E_max - E_min) <= E_t:
            print("Perceptron substitution")
            self.__type = NodeType.SUBSTITUTION
        else:
            # trained perceptron is acceptable
            print("Using trained perceptron")
            self.__nn.reinit_weights(old_weight)

        del preds #FIX memory issues


    def __create_split_node(self, lts: np.ndarray, X: np.ndarray):
        """
        This function create a split node in case the train stopping earlier (any improvement for a number of epochs)

        IDEA:
            * compute barycenter of two class with larger cardinality;
            * find hyperplane orthogonal passing the median point between the two barycenter.
        --------------

        :param lts: Local Training Set that will be split
        :param X: patterns which make predictions

        :return: return an arrays containing the two dataset split
        """
        y = lts[:, self.__num_features]

        zeros = X[np.where(y == 0)]
        ones = X[np.where(y == 1)]

        c_0 = self.__get_centroid(zeros)
        c_1 = self.__get_centroid(ones)

        del zeros, ones #FIX memory issues

        center = self.__get_center_between_centroids(c_0, c_1)
        split_rule = self.__split_hyperplane(c_0, c_1, center)

        self.__nn.reinit_weights(split_rule)

        lts0, lts1 = self.__dataset_split(lts, X)
        self.__is_splitted = True
        self.__type = NodeType.SPLIT

        del lts, X, c_0, c_1, center #FIX memory issues

        return lts0, lts1

    def __get_centroid(self, data: np.ndarray) -> np.ndarray:
        """
        :param data: dataset witch want to compute centroid

        :return: array containing centroid's coordinates
        """
        return np.mean(data, axis=0)

    def __get_center_between_centroids(self, c_1: np.ndarray, c_2: np.ndarray) -> np.ndarray:
        """
        This function calculates the center point between two point, in this case are centroids.
                            ((x1+x2+x3+...+xn)/2, (y1+y2+y3+...+yn)/2, ...)

        :param c_1: array containing centroid coordinates
        :param c_2: array containing centroid coordinates

        :return: array containing center coordinates
        """
        #center =
        return np.mean([c_1, c_2], axis=0)

    def __split_hyperplane(self, c_1: np.ndarray, c_2: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        This function compute orthogonal hyperplane to line passing through 2 points and passing for point P.

        IDEA:
            * compute orthogonal hyperplane to line passing through 2 points and passing for point P in this way:
                * compute direction_vector array, as a difference between two points: v = (B - A);
                * find a vector that is not collinear to the direction vector;
                * compute the normal vector of the line by taking the cross product of the direction vector and not_collinear vector;
                * find the known term d, forcing the hyperplane passing through P

            As not collinear vector we consider the cross product between direction vector and weight vector generated by perceptron

        :param c_1: centroid coordinates
        :param c_2: centroid coordinates
        :param p: median point between two centroids

        :return: array contains hyperplane's coefficients
        """
        direction_vector = np.subtract(c_2, c_1)

        return np.append(direction_vector, -p.dot(direction_vector))


    def __dataset_split(self, lts: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function create a local datasets starting from starting dataset stored in current node based on decision boundary

        IDEA:
            foreach patterns into dataset will compute the prediction and based on that the dataset will split

        --------------

        :param lts: Local Training Set that will be split
        :param X: patterns whitch make predictions

        :return: return an arrays containig the two dataset split
        """
        preds = self.__nn.predict(X)

        lts0 = lts[np.where(preds == 0)[0]]
        lts1 = lts[np.where(preds == 1)[0]]

        del preds #FIX memory issues

        return lts0, lts1

    def predict(self, X: np.ndarray):
        X = X.reshape(1, -1)

        if self.__type == NodeType.LEAF:
            return self.__label.value
        else:
            if self.__nn.predict(X) == 1.0:
                return self.__left.predict(X)
            else:
                return self.__right.predict(X)