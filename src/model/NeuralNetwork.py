from dataclasses import dataclass, field
from typing import Tuple, Any, List

from keras.callbacks import EarlyStopping, CSVLogger
#from Perceptron import Perceptron
#from sklearn.linear_model import Perceptron
from keras.layers import Dense
from keras.models import Sequential
from keras.activations import hard_sigmoid

import numpy as np

@dataclass
class NeuralNetwork:
    """
    Costruttore:
        NeuralNetwork(__num_input_features,
                      __num_classi = 2,
                      __num_perceptrons = 1)
    """
    #__perceptrons : np.ndarray = field(init=False)
    #__outputs : np.ndarray = field(init=False)
    __model : Sequential = field(init=False)
    __num_input_features: int
    __num_classi : int = 2
    __num_perceptrons: int = 1

    def set_num_classi(self, num_classi):
        self.__num_classi = num_classi

    def __post_init__(self):
        #if self.__num_classi == 2:
         #   self.__num_perceptrons = 1
        #else:
         #   self.__num_perceptrons = self.__num_classi

        #self.__perceptrons = np.empty(self.__num_perceptrons, dtype=Perceptron)
        self.__model = Sequential()

        self.__model.add(Dense(self.__num_perceptrons, input_shape=(self.__num_input_features,), activation=hard_sigmoid, kernel_initializer='glorot_uniform'))
        self.__model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #TODO: ampliare metriche

        #for i in range(self.__num_perceptrons):
         #   self.__perceptrons[i] = Perceptron(verbose=1, eta0=.1, early_stopping=True, n_iter_no_change=5, tol=1e-4) #TODO: mettere learning rate personalizzabile (eta0; )

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, wait_epochs: int, verbose: int = 0):
        #early stopping is useful for reduce training time and save time
        callbacks = [EarlyStopping(monitor="val_loss", patience=wait_epochs, verbose=1)]
        #callbacks = [EarlyStopping(monitor="val_loss", patience=wait_epochs, verbose=1),
         #            CSVLogger("log.csv", separator=',', append=False)]

        #hystory = self.__model.fit(X, y, epochs=epochs, verbose=verbose, validation_split=0.1, callbacks=callbacks)
        self.__model.fit(X, y, epochs=epochs, verbose=verbose, validation_split=0.1, callbacks=callbacks)
        #for perceptron in self.__perceptrons:
         #   model = perceptron.fit(X, y)

            #print(model.coef_)
            #print(model.intercept_)
            #print(model.n_iter_)
            #print(model.loss_function_)

            #print(f'\tMean Accuracy: {results.best_score_}')
            #print(f'\tConfig: {results.best_params_}')

            #print('\nModel parameters:')
        #print(f'\tWeights: {self.__model.weights}')
        #print(f'\tBias: {self.__model.get_bias()}\n')
            #print(f'\tN_iter: {model.n_iter_}')
            #print(f'\tN_weigth update: {model.t_}')
        #return hystory, self.__model.weights

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        #prediction = self.__model.predict(X, batch_size=1, verbose=0)
        #for perceptron in self.__perceptrons:
         #   output = perceptron.predict(X)
        return self.__model.predict(X, batch_size=X.shape[0], verbose=verbose)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        i: int = 0

        for perceptron in self.__perceptrons:
            print(f"Perceptron {i + 1}")
            i += 1
            print(f'\t{str}: {perceptron.score(X, y) * 100}')
        """
        return self.__model.evaluate(X, y, return_dict=True)

    def reinit_weights(self, weights: np.ndarray):
        bias = np.array([weights[self.__num_input_features - 1]])
        weights = weights[: self.__num_input_features].reshape((self.__num_input_features, 1))

        self.__model.set_weights([weights[: self.__num_input_features], bias])

    def get_weight(self):
        return self.__model.get_weights()

#-----------------------------------------------------------------------------------------------------------------------

def main() -> None:
    import pandas as pd
    #from sklearn.datasets import make_classification

    toy = pd.read_csv("toydata.csv").to_numpy()
    #toy = make_classification(10000, 40, n_classes=2)

    #X, y = toy[0], toy[1]
    X, y = toy[:, :2], toy[:, 2]

    shuffle_idx = np.arange(y.shape[0])
    shuffle_rng = np.random.RandomState(123)
    shuffle_rng.shuffle(shuffle_idx)
    X, y = X[shuffle_idx], y[shuffle_idx]

    X_train, X_test = X[shuffle_idx[:80]], X[shuffle_idx[80:]]
    y_train, y_test = y[shuffle_idx[:80]], y[shuffle_idx[80:]]

    # Normalize (mean zero, unit variance)
    #mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    #X_train = (X_train - mu) / sigma
    #X_test = (X_test - mu) / sigma

    nn = NeuralNetwork(X.shape[1])

    nn.fit(X_train, y_train, 500, 5)
    #print(nn.predict(X_test))

    #print(f"Training -> {nn.evaluate(X_train, y_train)}")
    #print(f"Testing -> {nn.evaluate(X_test, y_test)}")


    print(nn.get_weight())
    print(nn.get_weight()[0])
    print(nn.get_weight()[1])

    nn.reinit_weights(np.zeros(X.shape[1]))
    #print(statistics.history['loss'])
    print(nn.get_weight())
    print(nn.get_weight()[0])
    print(nn.get_weight()[1])

    #for pattern, expected_y in zip(X_test, y_test):
     #   print(f"{nn.predict(pattern.reshape(1, -1))} -> {expected_y}")

    #print("Train accuracy:")
    #nn.evaluate(X_train, y_train)

    #print("Test accuracy:")
    #nn.evaluate(X_test, y_test)


if __name__ == "__main__":
    main()

