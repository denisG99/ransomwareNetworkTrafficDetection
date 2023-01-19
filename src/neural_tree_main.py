import os
import sys
sys.path.insert(0, os.path.abspath('./model'))

import time
from model.NeuralTree import NeuralTree

EXPORTED_MODEL = "../exported_model"
DATASET_PATH = "../csv"

def main():
    try:
        with open(f"{EXPORTED_MODEL}/nt_unbalance.pkl", 'rb') as _:
            print("CARICAMENTO MODELLO DA ../exported_model/nt_unbalance.pkl")

            nt = NeuralTree.load_model(f"{EXPORTED_MODEL}/nt_unbalance.pkl")
    except IOError:
        nt = NeuralTree(f"{DATASET_PATH}/norm_train.csv", 25)
        start_time = time.time()

        print("INIZIO ADDESTRAMENTO NEURAL TREE")
        nt.train(250, 0)

        print(f"L'addestramento della bayesian network ha impiegato {time.time() - start_time} secondi")

        print("SALVATAGGIO MODELLO IN ../exported_model/nt_unbalance.pkl")
        nt.save_model(f"{EXPORTED_MODEL}/nt_unbalance.pkl")

if __name__ == "__main__":
    main()
