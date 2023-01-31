import os
import sys
sys.path.insert(0, os.path.abspath('./model'))

import pandas as pd
import time
import numpy as np

from scipy.stats import entropy

from model.NeuralTree import NeuralTree

EXPORTED_MODEL = "../exported_model"
DATASET_PATH = "../csv"

top_10 = ['Protocol', 'Bwd Header Len', 'Flow Iat Max', 'Bwd Pkts S', 'Flow Byts S', 'Timestamp', 'Flow Pkts S', 'Fwd Pkts S', 'Flow Duration', 'Src Ip', 'Label']

'''
top_20 = ['Pkt Len Var', 'Subflow Fwd Byts', 'Totlen Bwd Pkts', 'Pkt Len Std', 'Fwd Header Len.1', 'Fwd Seg Size Min', 'Tot Bwd Pkts', 'Fwd Iat Max', 'Fwd Header Len', 'Bwd Pkt Len Max', 'Protocol', 'Bwd Header Len', 'Flow Iat Max', 'Bwd Pkts S', 'Flow Byts S', 'Timestamp', 'Flow Pkts S', 'Fwd Pkts S', 'Flow Duration', 'Src Ip']
top_30 = ['Fwd Pkt Len Min', 'Dst Port', 'Flow Iat Mean', 'Pkt Len Min', 'Bwd Pkt Len Min', 'Dst Ip', 'Bwd Seg Size Avg', 'Subflow Bwd Byts', 'Bwd Pkt Len Mean', 'Fwd Iat Mean', 'Pkt Len Var', 'Subflow Fwd Byts', 'Totlen Bwd Pkts', 'Pkt Len Std', 'Fwd Header Len.1', 'Fwd Seg Size Min', 'Tot Bwd Pkts', 'Fwd Iat Max', 'Fwd Header Len', 'Bwd Pkt Len Max', 'Protocol', 'Bwd Header Len', 'Flow Iat Max', 'Bwd Pkts S', 'Flow Byts S', 'Timestamp', 'Flow Pkts S', 'Fwd Pkts S', 'Flow Duration', 'Src Ip']
'''

def main():
    try:
        with open(f"{EXPORTED_MODEL}/nt_unbalance.pkl", 'rb') as _:
            print("CARICAMENTO MODELLO DA ../exported_model/nt_unbalance.pkl")

            nt = NeuralTree.load_model(f"{EXPORTED_MODEL}/nt_unbalance.pkl")
    except IOError:
        dataset = pd.read_csv(f"{DATASET_PATH}/train.csv", low_memory=False)
        dataset = dataset[top_10].to_numpy()

        print(dataset.shape)

        labels, counts = np.unique(dataset[:, dataset.shape[1] - 1], return_counts=True)
        probs = counts / dataset.shape[0]

        nt = NeuralTree(entropy(probs, base=2), dataset.shape[1] - 1, dataset.shape[0], 25)
        #nt = NeuralTree(5)
        del probs, labels, counts

        start_time = time.time()

        print("INIZIO ADDESTRAMENTO NEURAL TREE")
        nt.train(dataset, 250, 1)

        print(f"L'addestramento della bayesian network ha impiegato {time.time() - start_time} secondi")

        print("SALVATAGGIO MODELLO IN ../exported_model/nt_unbalance.pkl")
        nt.save_model(f"{EXPORTED_MODEL}/nt_unbalance.pkl")

if __name__ == "__main__":
    main()
