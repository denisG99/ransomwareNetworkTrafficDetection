from dataclasses import dataclass, field
from typing import Tuple, Union, Dict, Any, Iterable

#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from scipy.stats import entropy

#from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork
from NodeType import NodeType
from Classification import Classification
import Node


import numpy as np
import uuid
#import cv2 as cv
import math

@dataclass
class Node:
    """
    Costruttore:
        Node(__entropy,
             __num_features,
             __cardinality,
             __label = Classification.NONE,
             __type = NodeType.DECISION,
             __threshold = 0.5,
             __num_classi = 2)
    """
    __id: str = field(init=False, repr=False) #node identifier used in logging
    __nn : NeuralNetwork = field(init=False, repr=False) #perceptron definition
    #__childs: np.ndarray = field(init=False) #list containing childs nodes
    __left: Node = field(init=False) #left child
    __right: Node = field(init=False) #rigth child
    __is_splitted: bool = field(init=False)  #indicate if node is split
    #__is_removed: bool = field(init=False)  #indicate if node has executed pattern removal
    #__patterns: np.ndarray  #field containing training data for perceptron
    __entropy: float # dataset entropy
    __num_features: int  #number of features into dataset
    __cardinality: int # dataset cardinality
    __label: Classification = Classification.NONE
    __type : NodeType = NodeType.DECISION
    __threshold : float = 0.5
    #__toler : float = 0.0 #tollerance that indicate the end of the branch training. Will be use in convergence test #TODO:forse da spostare
    __num_classi : int = 2

    # GETTER & SETTER
    def get_num_features(self) -> int:
        return self.__num_features

    #def get_childs(self) -> np.array:
     #   return self.__childs

    def get_left(self):
        return self.__left

    def get_right(self):
        return self.__right

    def get_type(self) -> NodeType:
        return self.__type

    def get_label(self):
        return self.__label

#-----------------------------------------------------------------------

    def __post_init__(self):
        self.__id = str(uuid.uuid4())
        self.__nn = NeuralNetwork(self.__num_features)
        #self.__entropy, occurs = self.__compute_entropy()
        #self.__childs = np.array([], dtype=Node)
        self.__left = None
        self.__right = None
        self.__is_splitted = False
        #self.__is_removed = False

        if self.__is_homogeneous(self.__cardinality):#self.__patterns[:, self.__num_features]):
            self.__type = NodeType.LEAF
            #self.__label = Classification(max(occurs, key=occurs.get))
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

        # print(self.__patterns)
        # print(counts)
        # print(labels)

        # if np.any(self.__patterns):
        #   return entropy(probs, base=2), np.zeros(2)
        # label = dict(zip(label, count))
        # print(probs)

        return entropy(probs, base=2), dict(zip(labels, counts))

    def __is_homogeneous(self, n) -> bool:
        """
        This function check if the dataset may be considered homogeneous. To do this you need to define some kind of
        threshold which depends on dataset cardinality.

        The threshold is thus defined:
                entropy_th = log2(n)/n, dove n è la cardinalità del dataset

        :return: true if the dataset is homogeneous, otherwise false
        """
        #n = self.__patterns.shape[0]

        #print(f"entropia = {self.__entropy}")
        #print(f"th_entropia = {math.log2(n) / n}")

        return self.__entropy <= (math.log2(n) / n)

    def train(self, data: np.ndarray, epochs: int, wait_epochs: int, verbose: int = 0) -> None:
        if not self.__type == NodeType.LEAF:
            X = data[:, 0 : self.__num_features]  # patterns dataset
            y = data[:, self.__num_features]  # labels dataset

            #self.__nn.fit(data, target, epochs, wait_epochs, verbose=verbose)
            self.__nn.fit(X, y, epochs, wait_epochs, verbose=verbose)

            #DATASET SPLITTING
            #split based on perceptron substitutions
            #preds = self.__nn.predict(data, verbose=verbose)
            preds = self.__nn.predict(X, verbose=verbose)
            print(f"Bias dopo addestrameto: {self.__nn.get_weight()[1]}")

            if not len(np.unique(preds)) == 1:
                #self.__make_acceptable_model(target, preds) #create trained perceptron that considered acceptable
                self.__make_acceptable_model(X, y, preds) #create trained perceptron that considered acceptable (perceptron substitution)
                print(f"Bias accettabile: {self.__nn.get_weight()[1]}")

                #lts0, lts1 = self.__dataset_split(self.__patterns, data, verbose=verbose)
                lts0, lts1 = self.__dataset_split(data, X, verbose=verbose)
            else: #split based on split node
                print("Creazione split rule")
                #lts0, lts1 = self.__create_split_node(self.__patterns, data, verbose=verbose)
                lts0, lts1 = self.__create_split_node(data, X, verbose=verbose)

            del data, X, y, preds #FIX for momory issues


            #print(lts0)
            #print(lts1)
            print(f"lts0: {np.unique(lts0[:, self.__num_features], return_counts=True)}")
            print(f"lts1: {np.unique(lts1[:, self.__num_features], return_counts=True)}")

            #CHILD CREATION + TRAINING
            #self.__left = Node(lts1, self.__num_features)
            #self.__right = Node(lts0, self.__num_features)

            #self.__left.train(epochs, wait_epochs, verbose=verbose)
            #self.__right.train(epochs, wait_epochs, verbose=verbose)

            entropy, occurs = self.__compute_entropy(lts1)
            self.__left = Node(entropy, self.__num_features, lts1.shape[0], Classification(max(occurs, key=occurs.get)))

            entropy, occurs = self.__compute_entropy(lts0)
            self.__right = Node(entropy, self.__num_features, lts0.shape[0], Classification(max(occurs, key=occurs.get)))

            self.__left.train(lts1, epochs, wait_epochs, verbose=verbose)
            self.__right.train(lts0, epochs, wait_epochs, verbose=verbose)

            print(f"END TRAINING Node {self.__id}")

        print(f"Node {self.__id} is {self.__type} Node (Label -> {self.__label})")

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
        #print(X)

        old_weight = self.__nn.get_weight()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #print(self.__nn.get_weight()[1])
        K_t = y_true.shape[0]

        #print(f"Perceptron addestrato: {tn, fp, fn, tp}")

        K_c = tn + tp
        E_0 = 1 - (K_c / K_t)

        # compute centroid of the local training set
        centroid = self.__get_centroid(X)

        #print(self.__nn.get_weight())

        # compute the new hyperplane passing throw centroid
        hyperplane = np.append(self.__nn.get_weight()[: self.__num_features], centroid.dot(self.__nn.get_weight()[: self.__num_features]))
        self.__nn.reinit_weights(hyperplane) # perceptron substitution
        #print(self.__nn.get_weight())

        preds = self.__nn.predict(X, verbose=1)

        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        #print(self.__nn.get_weight()[1])
        #print(f"Perceptron da valutare: {tn, fp, fn, tp}")

        K_c = tn + tp
        E_t = 1 - (K_c / K_t)

        K_ci = np.array([tp, tn])
        K_ti = np.array([tp + fp, tn + fn])

        E_i = 1 - np.divide(K_ci, K_ti)

        E_max, E_min = np.max(E_i), np.min(E_i)
        print(f"E_0 = {E_0}")
        print(f"E_0 / 2 = {E_0/2}")
        print(f"E_t = {E_t}")
        print(f"E_max - E_min = {E_max - E_min}")


        #TODO: ricontrollare condizione
        if E_t <= (E_0 / 2) and (E_max - E_min) <= E_t:
            print("Perceptron substitution")
            self.__type = NodeType.SUBSTITUTION
        else:
            # trained perceptron is acceptable
            print("Using trained perceptron")
            self.__nn.reinit_weights(old_weight)
        #print(self.__nn.get_weight())

        del preds #FIX memory issues


    def __create_split_node(self, lts: np.ndarray, X: np.ndarray, verbose: int = 0): #-> tuple[np.ndarray, np.ndarray, dict[Any, np.ndarray], np.ndarray]:
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
        #find the two larger cardinality class
        """
        largest_classes = self.__get_largest_class(lts, 2)
        splitted_by_class = dict()
        centroids = dict()

        for class_id in largest_classes:
            splitted_by_class[class_id] = self.__get_element_with_class(lts, class_id)

        for key in splitted_by_class.keys():
            centroids[key] = self.__get_centroid(splitted_by_class[key])

        # compute center between centroids
        center = self.__get_center_between_centroids(centroids)
        split_rule = self.__split_hyperplane(centroids, center)

        self.__nn.reinit_weights(split_rule)
        #print(self.__nn.get_weight())

        lts0, lts1 = self.__dataset_split(lts, X, verbose=verbose)
        self.__is_splitted = True

        return lts0, lts1
        """
        #lts.tofile("to_split.csv", ',')
        #np.savetxt("to_split.csv", lts, delimiter=',')
        y = lts[:, self.__num_features]

        #print(np.unique(y, return_counts=True))
        #print(y.shape)

        zeros = X[np.where(y == 0)]
        ones = X[np.where(y == 1)]

        #print(zeros)
        #print(zeros.shape)
        #print(ones)
        #print(ones.shape)

        c_0 = self.__get_centroid(zeros)
        c_1 = self.__get_centroid(ones)

        del zeros, ones #FIX memory issues

        center = self.__get_center_between_centroids(c_0, c_1)
        split_rule = self.__split_hyperplane(c_0, c_1, center)

        self.__nn.reinit_weights(split_rule)
        # print(self.__nn.get_weight())

        lts0, lts1 = self.__dataset_split(lts, X, verbose=verbose)
        self.__is_splitted = True

        del lts, X, c_0, c_1, center #FIX memory issues

        return lts0, lts1
    def __get_largest_class(self, lts: np.ndarray, num_largest: int = 2) -> np.ndarray:
        """
        This function count the cardinality foreach class e return the top n classes

        :param lts: Local Training Set that will be split
        :param num_largest: the first n the largest classes

        :return: array containing the largest classes
        """
        classes, counts = np.unique(lts[:, self.__num_features], return_counts=True)
        top_classes = list(sorted(zip(classes, counts), reverse=True))[:num_largest]
        largest_classes = np.zeros(num_largest)

        for i in range(num_largest):
            largest_classes[i] = top_classes[i][0]

        return largest_classes

    def __get_element_with_class(self, data: np.ndarray, class_id: int) -> np.ndarray:
        """
        :param data: dataset we want to split
        :param class_id: class that want to take

        :return: array containing record correspondent to a given class
        """
        return data[np.where(data[:, self.__num_features] == class_id)]

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
        #TODO: da verificare
        #classes = list(points.keys())

        direction_vector = np.subtract(c_2, c_1)
        #print(f"Differenza tra centroidi: {direction_vector}")

        return np.append(direction_vector, -p.dot(direction_vector))


    def __dataset_split(self, lts: np.ndarray, X: np.ndarray, verbose: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function create a local datasets starting from starting dataset stored in current node based on decision boundary

        IDEA:
            foreach patterns into dataset will compute the prediction and based on that the dataset will split

        --------------

        :param lts: Local Training Set that will be split
        :param X: patterns whitch make predictions

        :return: return an arrays containig the two dataset split
        """
        #lts0 = np.array([])
        #lts1 = np.array([])

        preds = self.__nn.predict(X, verbose=verbose)
        #status = 1
        #print(preds)

        #data = np.append(lts, preds, axis=1)

        """
        for pred, row in zip(preds, lts):
            #print(pattern)
            #print(row)
            print(f"{status}/{preds.shape[0]}")
            if pred >= 0.5:
                lts1 = np.append(lts1, row)
            else:
                lts0 = np.append(lts0, row)

            status += 1
        """
        lts0 = lts[np.where(preds == 0)[0]]
        lts1 = lts[np.where(preds == 1)[0]]
        #lts0 = np.resize(lts0, (int(len(lts0) / (self.__num_features + 1)), self.__num_features + 1))
        #lts1 = np.resize(lts1, (int(len(lts1) / (self.__num_features + 1)), self.__num_features + 1))

        #print(preds.shape)
        #print(np.unique(preds, return_counts=True))
        #print(f"lts0:\n {lts0}")
        #print(f"lts1:\n {lts1}")

        #row0, row1 = lts0.shape[0], lts1.shape[0]

        del preds #FIX memory issues

        #return lts0.reshape(row0, self.__num_features + 1), lts1.reshape(row1, self.__num_features + 1)
        return lts0, lts1

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        #TODO: fare in modo che funzioni su input composto da moltepici sample

        if self.__type == NodeType.LEAF:
            return self.__label.value
        else:
            if self.__nn.predict(X) == 1:
                #return self.__childs[0].predict(X, verbose=verbose)
                return self.__left.predict(X, verbose=verbose)
            else:
                #return self.__childs[1].predict(X, verbose=verbose)
                return self.__right.predict(X, verbose=verbose)

    #TODO: refactoring della funzione in modo tale da farla più elegnate
    #FUNCTION DOESN'T WORK
   # def visualize_node(self, img, height: int, width: int, dim: int, is_left: bool = False, is_right: bool = False, level: int = 0) -> np.ndarray:
        """
        This function create a visualization of Node that follow these rules:
            * white rectangle -> decision node
            * blue rectangle -> split node
            * red circle -> leaf node labeled as malware
            * green circle -> leaf node labeled as benign

        :param img: image
        :param height: image height
        :param width: image width
        :param dim: for circle indicate radius instead for rectangle indicate length of square side
        :param is_left: indicate if the node that I draw is left child
        :param is_right: indicate if the node that I draw is right child
        :param level: level of the tree (maintain the default value, otherwise it draws the tree lower in resulting image)
        """
        """
        offset = 10
        #offset used in drawing child
        x_offset = (dim // 2)
        y_offset = offset + dim + (dim // 2)

        if not self.__left is None and not self.__right is None:
            if self.get_type() == NodeType.LEAF:
                if self.get_label() == Classification.MALWARE:
                    if is_left and not is_right:
                        img = cv.circle(img,  # image
                                        (((width // 2) - (x_offset * level)), (((dim // 2) + offset) * level)),  # center_coordinates
                                        (dim // 2),  # radius
                                        (0, 0, 255),  # color
                                        -1)  # thickness
                    elif is_right and not is_left:
                        img = cv.circle(img,  # image
                                        (((width // 2) + (x_offset * level)), (((dim // 2) + offset) * level)),  # center_coordinates
                                        (dim // 2),  # radius
                                        (0, 0, 255),  # color
                                        -1)  # thickness
                    else:
                        img = cv.circle(img,  # image
                                        (width // 2, ((dim // 2) + offset)),  # center_coordinates
                                        (dim // 2),  # radius
                                        (0, 0, 255),  # color
                                        -1)  # thickness
                else:
                    if is_left and not is_right:
                         img = cv.circle(img, #image
                                          (((width // 2) - (x_offset * level)), (((dim // 2) + offset) * level)),  # center_coordinates
                                          (dim // 2), #radius
                                          (0, 255, 0), #color
                                          -1) #thickness
                    elif is_right and not is_left:
                        img = cv.circle(img,  # image
                                        (((width // 2) + (x_offset * level)), (((dim // 2) + offset) * level)),  # center_coordinates
                                        (dim // 2),  # radius
                                        (0, 255, 0),  # color
                                        -1)  # thicknessù
                    else:
                        img = cv.circle(img,  # image
                                        (width // 2, ((dim // 2) + offset)),  # center_coordinates
                                        (dim // 2),  # radius
                                        (0, 255, 0),  # color
                                        -1)  # thickness

            elif self.get_type() == NodeType.DECISION:
                if is_left and not is_right:
                    img = cv.rectangle(img, #image
                                        ((width // 2) - (dim // 2) + (x_offset * level), offset + (y_offset * level)), #start_point
                                        ((width // 2) + (dim // 2) + (x_offset * level), dim + offset + (y_offset * level)), #end_point
                                        (0, 0, 0), #color
                                        2) #thickness
                elif is_right and not is_left:
                    img = cv.rectangle(img,  # image
                                       ((width // 2) - (dim // 2) + (x_offset * level), offset + (y_offset * level)),  # start_point
                                       ((width // 2) + (dim // 2) + (x_offset * level), dim + offset + (y_offset * level)),  # end_point
                                       (0, 0, 0),  # color
                                       2)  # thickness
                else:
                    img = cv.rectangle(img,  # image
                                       ((width // 2) - (dim // 2), offset),  # start_point
                                       ((width // 2) + (dim // 2), dim + offset),  # end_point
                                       (0, 0, 0),  # color
                                       2)  # thickness

            elif self.get_type() == NodeType.SPLIT:
                if is_left and not is_right:
                    img = cv.rectangle(img,  # image
                                       ((width // 2) - (dim // 2) + (x_offset * level), offset + (y_offset * level)), # start_point
                                       ((width // 2) + (dim // 2) + (x_offset * level), dim + offset + (y_offset * level)),  # end_point
                                       (0, 0, 0),  # color
                                       -1)  # thickness
                elif is_right and not is_left:
                    img = cv.rectangle(img,  # image
                                       ((width // 2) - (dim // 2) + (x_offset * level), offset + (y_offset * level)), # start_point
                                       ((width // 2) + (dim // 2) + (x_offset * level), dim + offset + (y_offset * level)),  # end_point
                                       (0, 0, 0),  # color
                                       -1)  # thickness
                else:
                    img = cv.rectangle(img,  # image
                                       ((width // 2) - (dim // 2), offset),  # start_point
                                       ((width // 2) + (dim // 2), dim + offset),  # end_point
                                       (0, 0, 0),  # color
                                       -1)  # thickness

            if is_left and not is_right:
                #draw left child line
                img = cv.line(img, #image
                               ((width // 2) - (x_offset * level), offset + dim + (y_offset * level)), #start_point
                               ((width // 2) - (dim // 2) - (x_offset * level), offset + dim + (dim // 2) + (y_offset * level)), #end point
                               (0, 0, 0), #color
                               2) #thickness

                #draw right child line
                img = cv.line(img,  # image
                              ((width // 2) - (x_offset * level), offset + dim + (y_offset * level)),  # start_point
                              ((width // 2) + (dim // 2) - (x_offset * level), offset + (dim // 2) + dim + (y_offset * level)), #end_point
                              (0, 0, 0),  # color
                              2)  # thickness
            elif is_right and not is_left:
                # draw left child line
                img = cv.line(img,  # image
                              ((width // 2) + (x_offset * level), offset + dim + (y_offset * level)),  # start_point
                              ((width // 2) + (dim // 2) + (x_offset * level), offset + dim + (dim // 2) + (y_offset * level)),
                              # ((width // 2), dim + offset ), #end_point
                              (0, 0, 0),  # color
                              2)  # thickness

                # draw right child line
                img = cv.line(img,  # image
                              ((width // 2) + (x_offset * level), offset + dim + (y_offset * level)),  # start_point
                              ((width // 2) + (dim // 2) + (x_offset * level), offset + (dim // 2) + dim + (y_offset * level)),  # end_point
                              (0, 0, 0),  # color
                              2)  # thickness
            else:
                # draw left child line
                img = cv.line(img,  # image
                              ((width // 2), offset + dim),  # start_point
                              ((width // 2) - (dim // 2), offset + dim + (dim // 2)),
                              # ((width // 2), dim + offset ), #end_point
                              (0, 0, 0),  # color
                              2)  # thickness

                # draw right child line
                img = cv.line(img,  # image
                              ((width // 2), offset + dim),  # start_point
                              ((width // 2) + (dim // 2), offset + (dim // 2) + dim),  # end_point
                              (0, 0, 0),  # color
                              2)  # thickness

            level += 1
            #img = self.__left.visualize_node(img, height, width, dim, is_left=True, is_right=False, level=level)
            img= self.__right.visualize_node(img, height, width, dim, is_left=False, is_right=True, level=level)

        return img
    """
#-----------------------------------------------------------------------------------------------------------------------

def main() -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    from sklearn.datasets import make_circles
    from sklearn.datasets import make_classification
    import numpy as np


    patterns = pd.read_csv("../../csv/train.csv").to_numpy()
    #print(patterns)
    #print(patterns.shape)
    #X, y = make_moons(100)
    #X, y = make_classification(10000, 40, n_classes=2)


    #print(X)
    #y= np.reshape(y, (10000, 1))

    #X_train, X_test, y_train, y_test = train_test_split(patterns[:, :2], patterns[:, 2], test_size=.3, train_size=.7)

    #patterns = np.append(X_train, np.reshape(y_train, (-1, 1)), axis=1)

    root = Node(patterns, patterns.shape[1] - 1)
    #X, y = patterns[0], patterns[1]

    #print(root)

    root.train(250, 5, verbose=1)
    #lts0, lts1, centroids, center = root.create_split_node(patterns, X)
    #print(root)

    #lts0, lts1 = root.dataset_split(patterns, patterns[:, 0: patterns.shape[1] - 1])

    #print(lts0)
    #print(lts1)

    #print(centroids)
    #split_rule = root.split_hyperplane(centroids, center)

    #x = np.linspace(-3, 3, 10)

    #plotting
    """
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='class 0', marker='o')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='class 1', marker='s')
    plt.title('Dataset')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    #plt.scatter(centroids[0.0][0], centroids[0.0][1], label='centroid class 0', marker='o')
    #plt.scatter(centroids[1.0][0], centroids[1.0][1], label='centroid class 1', marker='s')
    #plt.scatter(center[0], center[1], label='centroid median', marker='^')
    #plt.axline(centroids[0.0], centroids[1.0])
    #plt.plot(x, -((split_rule[0] * x - split_rule[2]) / split_rule[1]), '-r')
    plt.legend()
    plt.show()
    
    #print(root)
    print(root.predict(np.zeros(40).reshape((1, -1))))
    
    #lts0, lts1 = root.dataset_split(patterns, X)
    #root.dataset_split(patterns, X)
    
    #X0, y0, X1, y1 = lts0[:, :2], lts0[:, 2], lts1[:, :2], lts1[:, 2]

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
