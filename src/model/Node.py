from dataclasses import dataclass, field
from typing import Tuple, Union, Dict, Any, Iterable

from sklearn.model_selection import train_test_split
from scipy.stats import entropy

from matplotlib import pyplot as plt

from NeuralNetwork import NeuralNetwork
from NodeType import NodeType
from Classification import Classification
import Node


import numpy as np
import uuid
import cv2 as cv

ENTROPY_TH: float = 0.1 #TODO: da cambiare

@dataclass
class Node:
    """
    Costruttore:
        Node(__patterns,
             __num_features,
             __label = Classification.NONE,
             __type = NodeType.DECISION,
             __threshold = 0.5,
             __num_classi = 2)
    """
    __id: str = field(init=False, repr=False) #node identifier used in logging
    __nn : NeuralNetwork = field(init=False, repr=False) #perceptron definition
    __entropy: float = field(init=False)  #dataset entropy
    #__childs: np.ndarray = field(init=False) #list containing childs nodes
    __left: Node = field(init=False) #left child
    __right: Node = field(init=False) #rigth child
    __is_splitted: bool = field(init=False)  #indicate if node is split
    #__is_removed: bool = field(init=False)  #indicate if node has executed pattern removal
    __patterns: np.ndarray  #field containing training data for perceptron
    __num_features: int  #number of features into dataset
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
        self.__entropy, occurs = self.__compute_entropy()
        #self.__childs = np.array([], dtype=Node)
        self.__left = None
        self.__right = None
        self.__is_splitted = False
        #self.__is_removed = False

        if self.__entropy <= ENTROPY_TH:
            self.__type = NodeType.LEAF
            self.__label = Classification(max(occurs, key=occurs.get))
            self.__num_classi = 1
            self.__nn.set_num_classi(1)

    def __compute_entropy(self) -> Tuple[Union[float, np.ndarray, Iterable, int], Dict[Any, Any]] or Tuple[None, None]:
        probs = list()
        label = self.__patterns[:, self.__num_features]
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


    def train(self, epochs: int, wait_epochs: int, plot = None, verbose: int = 0) -> None:
        #print(self.__patterns)
        #print(self.__type)
        #print(f"START TRAINING Node {self.__id}")

        if not self.__type == NodeType.LEAF:
            #TODO da togliere la divisione in train e test set
            data = self.__patterns[:, 0 : self.__num_features]  # patterns dataset
            target = self.__patterns[:, self.__num_features]  # labels dataset
            #X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=.2,train_size=.8)  # dataset split with 80-20 splitting rule
            #classes, counts = np.unique(target, return_counts=True)

            effective_epochs, weights = self.__nn.fit(data, target, epochs, wait_epochs, verbose=verbose)

            #plt.plot([-3, 3], [((-(weights[0][0] * (-3) + weights[1][0])) / weights[0][1]),
             #                      ((-(weights[0][0] * 3 + weights[1][0])) / weights[0][1])])
            #self.__nn.evaluate(X_train, y_train, "Train accuracy")
            #self.__nn.evaluate(X_test, y_test, "Test accuracy")

            lts0, lts1 = self.__dataset_split(self.__patterns, data, verbose=verbose)

            #print(np.any(lts0))
            #print(np.any(lts1))

            if not np.any(lts0) or not np.any(lts1):
                #print("Creazione split rule")
                lts0, lts1, _, _ = self.__create_split_node(self.__patterns, data, verbose=verbose)

            #goodware_lts = self.__pattern_removal(goodware_lts)
            #malware_lts= self.__pattern_removal(malware_lts)
            #self.__childs = np.append(self.__childs, [Node(lts1, self.__num_features),
             #                                         Node(lts0, self.__num_features)])
            self.__left = Node(lts1, self.__num_features)
            self.__right = Node(lts0, self.__num_features)

            #print(f"\tNode {self.__id} is {self.__type} Node (Label -> {self.__label})")

            #else:
             #   lts0, lts1, _, _ = self.__create_split_node(self.__patterns, data)

                #lts0 = self.__pattern_removal(lts0)
                #lts1 = self.__pattern_removal(lts1)

                #self.__childs = np.append(self.__childs, [Node(lts1, self.__num_features),
                 #                                         Node(lts0, self.__num_features)])

            #print(goodware_lts)
            #print(malware_lts)

            #for child in self.__childs:
                #child.train(epochs, wait_epochs, plot, verbose=verbose)
            self.__left.train(epochs, wait_epochs, plot, verbose=verbose)
            self.__right.train(epochs, wait_epochs, plot, verbose=verbose)

            #print(f"END TRAINING Node {self.__id}")

            print(f"VALUTAZIONE NODO {self.__id}")
            print(f"\tTraining -> {self.__nn.evaluate(X_train, y_train)}")
            print(f"\tTesting -> {self.__nn.evaluate(X_test, y_test)}")

        print(f"Node {self.__id} is {self.__type} Node (Label -> {self.__label})")

        return plot

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
        #TODO: da sistemare il fatto che iperpiano non passa per il punto centrale e non è ortogonale alla retta passante per i due centroidi
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

        return lts0, lts1, centroids, center

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
        centroid_cords = np.zeros(self.__num_features)

        for i in range(self.__num_features):
            centroid_cords[i] = np.sum(data[:, i]) / len(data[:, i])

        return centroid_cords

    def __get_center_between_centroids(self, centroids: dict) -> np.ndarray:
        """
        This function calculates the center point between two point, in this case are centroids.
                            ((x1+x2+x3+...+xn)/2, (y1+y2+y3+...+yn)/2, ...)

        :param centroids: array containing centroids coordinates

        :return: array containing center coordinates
        """
        center = np.zeros(self.__num_features)

        for key in centroids.keys():
            for i in range(self.__num_features):
                center[i] += centroids[key][i]

        return center / 2

    def __split_hyperplane(self, points: dict, p: np.ndarray) -> np.ndarray:
        """
        This function compute orthogonal hyperplane to straight passing through 2 points, passing for point P.

        IDEA:
            * compute hyperplane through 2 centroids in this way:
                * compute direction array, as a difference between two points: v = (B - A)
            * set the proportionality of two vector define as n = a*v, with a = 1 so n = v
            * find the known term d, forcing the hyperplane passing through P

        :param points: centroids list
        :param p: median point between two centroids

        :return: array contains hyperplane's coefficients
        """
        #TODO: da sistemare il fatto che iperpiano non passa per il punto centrale e non è ortogonale alla retta passante per i due centroidi
        #hyperplane = np.array([])
        classes = list(points.keys())

        hyperplane = np.subtract(points.get(classes[1]), points.get(classes[0]))
        hyperplane = np.append(hyperplane, (-1 * sum(hyperplane * p)))

        return hyperplane


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
        lts0 = np.array([])
        lts1 = np.array([])

        for pattern, row in zip(X, lts):
            #print(pattern)
            #print(row)
            if self.__nn.predict(pattern.reshape(1, -1), verbose=verbose) >= 0.5:
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

    def predict(self, X, verbose: int = 0) -> Classification:
        if self.__type == NodeType.LEAF:
            return self.__label
        else:
            if self.__nn.predict(X) >= self.__threshold:
                #return self.__childs[0].predict(X, verbose=verbose)
                return self.__left.predict(X, verbose=verbose)
            else:
                #return self.__childs[1].predict(X, verbose=verbose)
                return self.__right.predict(X, verbose=verbose)

    #TODO: refactoring della funzione in modo tale da farla più elegnate
    #FUNCTION DOESN'T WORK
    def visualize_node(self, img, height: int, width: int, dim: int, is_left: bool = False, is_right: bool = False, level: int = 0) -> np.ndarray:
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

#-----------------------------------------------------------------------------------------------------------------------

def main() -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    from sklearn.datasets import make_circles
    from sklearn.datasets import make_classification
    import numpy as np


    #patterns = pd.read_csv("toydata.csv", header=None).to_numpy()
    #print(patterns.shape)
    #X, y = make_moons(100)
    X, y = make_classification(10000, 40, n_classes=2)


    #print(X)
    y= np.reshape(y, (10000, 1))
    patterns = np.append(X, y, axis=1)

    root = Node(patterns, patterns.shape[1] - 1)
    #X, y = patterns[:, :2], patterns[:, 2]
    #X, y = patterns[0], patterns[1]

    root.train(250, 5, plt.plot())
    #lts0, lts1, centroids, center = root.create_split_node(patterns, X)
    #print(root)

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
    """
    #print(root)
    print(root.predict(np.zeros(40).reshape((1, -1))))
    """
    lts0, lts1 = root.dataset_split(patterns, X)
    #root.dataset_split(patterns, X)
    """
    #X0, y0, X1, y1 = lts0[:, :2], lts0[:, 2], lts1[:, :2], lts1[:, 2]

    """
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