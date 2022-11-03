from dataclasses import dataclass, field
from Neuron import Neuron

import numpy as np

@dataclass
class Perceptron:
    __neuron : Neuron = field(default_factory=Neuron, init=False)
    #__inputs: np.ndarray = field(init=False)
    __num_features: int
    __weights: np.ndarray = field(init=False)
    __learning_rate : float = 0.1
    __bias: float = 0.0

    def __post_init__(self):
        self.__weights = np.zeros(self.__num_features) #TODO: realizzare algoritmo migliore per la scelta dei pesi iniziali

        # if the learning rate value is not between 0 and 1, setting to default value (0.1)
        if not 0 <= self.__learning_rate <= 1:
            self.__learning_rate = 0.1

    def get_weigths(self):
        return self.__weights

    def get_bias(self):
        return self.__bias

    def fit(self, X, y, epochs):
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(epochs):

            for idx, x_i in enumerate(X):
                #linear_output = self.__neuron.weighted_sum(self.__weights, self.__bias, x_i)
                #y_predicted = self.__neuron.af(linear_output)
                y_predicted = self.predict(x_i)

                # Perceptron update rule
                update = self.__learning_rate * (y_[idx] - y_predicted)

                self.__weights += update * x_i
                self.__bias += update

    def predict(self, X):
        linear_output = self.__neuron.weighted_sum(self.__weights, self.__bias, X)

        return self.__neuron.af(linear_output)

    def evaluate(self, X, y):
        predictions = self.predict(X)

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

    print('Model parameters:')
    print(f'\tWeights: {perceptron.get_weigths()}')
    print(f'\tBias: {perceptron.get_bias()}\n')

    print(f'Train accuracy: {perceptron.evaluate(X_train, y_train) * 100}')
    print(f'Test accuracy: {perceptron.evaluate(X_test, y_test) * 100}')

if __name__ == "__main__":
    main()