import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import pandas as pd

EXPORTED_MODEL = "../exported_model"
DATASET_PATH = "../csv"

def main():
    training_set = pd.read_csv(f"{DATASET_PATH}/train.csv", low_memory=False)
    #test_set = pd.read_csv(f"{DATASET_PATH}/test.csv", low_memory=False)

    X_train, y_train = training_set.loc[:, 'Src Ip' : 'Idle Min'], training_set.loc[:, 'Label']
    #X_test, y_test = test_set.loc[:, 'Src Ip' : 'Idle Min'], test_set.loc[:, 'Label']

    print("DATASET ACQUISITI")

    del training_set#, test_set

    with open(f"{EXPORTED_MODEL}/knn_unbalance.pkl", 'rb') as file:
        knn_model = pkl.load(file)

    importance = permutation_importance(knn_model, X_train, y_train)

    importance = importance['importances_mean']

    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))

    #plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

if __name__ == "__main__":
    main()