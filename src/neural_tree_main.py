import time
from model.NeuralTree import NeuralTree

EXPORTED_MODEL = "../exported_model"
DATASET_PATH = "../csv"

def main():
    try:
        with open(f"{EXPORTED_MODEL}/nt_unbalance.pkl", 'rb') as file:
            print("CARICAMENTO MODELLO DA ../exported_model/nt_unbalance.pkl")

            nt = NeuralTree.load_model(f"{EXPORTED_MODEL}/nt_unbalance.pkl")
    except IOError:
        nt = NeuralTree(f"{DATASET_PATH}/train.csv", 5)
        start_time = time.time()

        print("INIZIO ADDESTRAMENTO NEURAL TREE")
        nt.train(250, 1)#fit(X_train_balanced, y_train_balanced)

        print(f"L'addestramento della bayesian network ha impiegato {time.time() - start_time} secondi")

        print("SALVATAGGIO MODELLO IN ../exported_model/nt_unbalance.pkl")
        nt.save_model(f"{EXPORTED_MODEL}/nt_unbalance.pkl")

if __name__ == "__main__":
    main()