from dataclasses import dataclass, field
from typing import Tuple

from sklearn.metrics import precision_score
from numpy.random import RandomState

from Neuron import Neuron

import numpy as np

@dataclass
class Perceptron:
    #fields
    __neuron : Neuron = field(default_factory=Neuron, init=False)
    __num_features: int
    __weights: np.ndarray = field(init=False)
    __learning_rate : float = 0.1
    __bias: float = 1.0
    __threshold: float = 0.5

    def __post_init__(self):
        self.__weights, self.__bias = self.__init_weights()

        # if the learning rate value is not between 0 and 1, setting to default value (0.1)
        if not 0 <= self.__learning_rate <= 1:
            self.__learning_rate = 0.1

    def get_weigths(self) -> np.ndarray:
        return self.__weights

    def get_bias(self) -> float:
        return self.__bias

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(epochs):
            for idx, x_i in enumerate(X):
                #linear_output = self.__neuron.weighted_sum(self.__weights, self.__bias, x_i)
                #y_predicted = self.__neuron.af(linear_output)
                y_predicted = self.predict(x_i)

                #print(f"{y_predicted} -> {y_[idx]}")

                # Perceptron update rule
                update = self.__learning_rate * (y_[idx] - y_predicted)

                self.__weights += update * x_i
                self.__bias += update

    def predict(self, X: np.ndarray) -> float:
        linear_output = self.__neuron.weighted_sum(self.__weights, self.__bias, X)

        return self.__neuron.af(linear_output)

    def __init_weights(self) -> Tuple[np.ndarray, float]:  # Glorot uniform
        nin: int = self.__num_features + 1
        nout: int = 1
        sd = np.sqrt(6.0 / (nin + nout))
        weigths: np.ndarray = np.empty(nin - 1, dtype=np.float32 )#np.array({}, dtype=np.float32, )

        for i in range(nin - 1):
            weigths[i] = np.float32(RandomState().uniform(-sd, sd))

        bias = float(RandomState().uniform(-sd, sd))

        return weigths, bias

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        predictions = np.where(predictions >= self.__threshold, 1, 0)

        accuracy = np.sum(y == predictions) / len(predictions)
        return accuracy

#-----------------------------------------------------------------------------------------------------------------------

def main() -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    #dataset preparation
    toy = pd.read_csv("toydata.csv").to_numpy()

    X, y = toy[:, :2], toy[:, 2]

    shuffle_idx = np.arange(y.shape[0])
    shuffle_rng = np.random.RandomState(123)
    shuffle_rng.shuffle(shuffle_idx)
    X, y = X[shuffle_idx], y[shuffle_idx]

    X_train, X_test = X[shuffle_idx[:70]], X[shuffle_idx[70:]]
    y_train, y_test = y[shuffle_idx[:70]], y[shuffle_idx[70:]]

    # Normalize (mean zero, unit variance)
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    """""
    #plotting
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label='class 0', marker='o')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], label='class 1', marker='s')
    plt.title('Training set')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.legend()
    plt.show()
    """""

    perceptron = Perceptron(2)

    perceptron.fit(X_train, y_train, epochs=5)

    print('\nModel parameters:')
    print(f'\tWeights: {perceptron.get_weigths()}')
    print(f'\tBias: {perceptron.get_bias()}\n')

    print(f'Train accuracy: {perceptron.evaluate(X_train, y_train) * 100}')
    print(f'Test accuracy: {perceptron.evaluate(X_test, y_test) * 100}')

    #plotting decision boundaries
    w, b = perceptron.get_weigths(), perceptron.get_bias()

    x0_min = -2
    x1_min = ((-(w[0] * x0_min) - b)
              / w[1])

    x0_max = 2
    x1_max = ((-(w[0] * x0_max) - b)
              / w[1])

    # x0*w0 + x1*w1 + b = 0
    # x1  = (-x0*w0 - b) / w1

    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(7, 3))

    ax[0].plot([x0_min, x0_max], [x1_min, x1_max])
    ax[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label='class 0', marker='o')
    ax[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], label='class 1', marker='s')

    ax[1].plot([x0_min, x0_max], [x1_min, x1_max])
    ax[1].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], label='class 0', marker='o')
    ax[1].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], label='class 1', marker='s')

    ax[1].legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()