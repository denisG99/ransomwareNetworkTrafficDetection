from dataclasses import dataclass, field

from sklearn.linear_model import Perceptron

import numpy as np

@dataclass
class NeuralNetwork:
    """
    Costruttore:
        NeuralNetwork(__num_input_features,
                      __num_classi = 2,
                      __num_perceptrons = 1)
    """
    __model : Perceptron = field(init=False)
    __num_input_features: int
    __num_classi : int = 2
    __num_perceptrons: int = 1

    def set_num_classi(self, num_classi):
        self.__num_classi = num_classi

    def __post_init__(self):
        self.__model = Perceptron(max_iter=250, verbose=0, n_iter_no_change=5, eta0=0.1, tol=0.0001)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.__model.fit(X, y, coef_init= self.__glorot_uniform(self.__num_input_features, self.__num_perceptrons), intercept_init=1)

    def __glorot_uniform(self, fan_in, fan_out) -> np.ndarray:
        """
        Compute Glorot-Xavier weight initializer with uniform distibution

        :param fan_in: input nodes
        :param fan_out: output nodes depending on a single output node
        :return: array containing weight
        """
        scale = np.sqrt(6.0 / (fan_in + fan_out))

        return np.random.uniform(low=-scale, high=scale, size=self.__num_input_features)


    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.__model.predict(X)

    def reinit_weights(self, weights: np.ndarray) -> None:
        bias = np.array(weights[self.__num_input_features])
        weights = weights[: self.__num_input_features].reshape((1,self.__num_input_features))

        self.__model.coef_ = weights
        self.__model.intercept_ = bias

    def get_weight(self) -> np.ndarray:
        return np.append(self.__model.coef_, self.__model.intercept_)