import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np

EXPORTED_MODEL = "../exported_model"
DATASET_PATH = "../csv"


def get_centroid(data: np.ndarray) -> np.ndarray:
    """
    :param data: dataset witch want to compute centroid

    :return: array containing centroid's coordinates
    """
    return np.mean(data, axis=0)

def get_center_between_centroids(c_1: np.ndarray, c_2: np.ndarray) -> np.ndarray:
    #return np.add(c_1, c_2) / 2
    return np.mean([c_1, c_2], axis=0)

def split_hyperplane(c_1: np.ndarray, c_2: np.ndarray, p: np.ndarray) -> np.ndarray:
    direction_vector = np.subtract(c_2, c_1)

    return np.append(direction_vector, -p.dot(direction_vector))

def main():
    data = pd.read_csv("./model/toydata.csv", header=None).to_numpy()

    X, y = data[:,:2], data[:, 2]

    zeros = X[np.where(y==0)]
    ones = X[np.where(y==1)]

    c_0 = get_centroid(zeros)
    c_1 = get_centroid(ones)
    center = get_center_between_centroids(c_0, c_1)
    split_rule = split_hyperplane(c_0, c_1, center)

    print(split_rule)

    x = np.linspace(-5, 5, 10)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='class 0', marker='o')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='class 1', marker='s')
    plt.title('Dataset')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.scatter(c_0[0], c_0[1], marker='o')
    plt.scatter(c_1[0], c_1[1], marker='s')
    plt.scatter(center[0], center[1], marker='^')
    plt.axline(c_0, c_1)
    plt.plot(x, -((split_rule[0] * x + split_rule[2]) / split_rule[1]), '-r')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()