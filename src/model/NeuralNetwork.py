from dataclasses import dataclass, field

from Perceptron import Perceptron

import numpy as np

@dataclass
class NeuralNetwork:
    __perceptrons : np.ndarray = field(init=False)
    __outputs : np.ndarray = field(init=False)
    __num_input_features: int
    __num_classi : int = 2

    def __post_init__(self):
        self.__outputs = np.zeros(self.__num_classi)
        self.__perceptrons = np.empty(self.__num_classi, dtype=Perceptron)

        for i in range(self.__num_classi):
            self.__perceptrons[i] = Perceptron(self.__num_input_features)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        for perceptron in self.__perceptrons:
            perceptron.fit(X, y, epochs)

            #print('\nModel parameters:')
            #print(f'\tWeights: {perceptron.get_weigths()}')
            #print(f'\tBias: {perceptron.get_bias()}\n')

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> None:
        i: int = 0

        for perceptron in self.__perceptrons:
            print(f"\tPerceptron {i}")
            i += 1
            print(f'\t\t{perceptron.evaluate(X, y)}')


#-----------------------------------------------------------------------------------------------------------------------

def main() -> None:
    import pandas as pd

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

    nn = NeuralNetwork(2)

    nn.fit(X_train, y_train, 5)

    print("Train accuracy:")
    nn.evaluate(X_train, y_train)

    print("Test accuracy:")
    nn.evaluate(X_test, y_test)


if __name__ == "__main__":
    main()