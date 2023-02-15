from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np

def main():
    #importing data
    data = pd.read_csv("../csv/train.csv")
    X, y = data.loc[:, "Src Ip" : "Idle Min"], data.loc[:, "Label"]
    del data

    #inizialize model
    perceptron = Perceptron()

    #hyper-parametrization
    parameters = {'max_iter' : [250, 500, 750, 1000],
                  'tol': [1e-4, 1e-3, 1e-2, 0.1],
                  'eta0': [0.1, 0.15, 0.2, 0.25],
                  'n_iter_no_change': [5, 10, 15, 20]
                  } #dictionary contains parameters that I want to test

    perceptron_grid = GridSearchCV(estimator=perceptron, param_grid=parameters, cv=5, n_jobs=-1).fit(X, y)

    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",perceptron_grid.best_estimator_)
    print("\n The best score across ALL searched params:\n",perceptron_grid.best_score_)
    print("\n The best parameters across ALL searched params:\n",perceptron_grid.best_params_)

if __name__ == "__main__":
    main()