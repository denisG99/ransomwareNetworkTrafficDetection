"""
This file contain the script file for testing Neural Tree
"""

from NeuralTree import NeuralTree

import numpy as np
import time

def main() -> None:
    start_time = time.time()
    nt = NeuralTree("/Users/denisgasparollo/PycharmProjects/ransomwareNetworkTrafficDetection/src/model/toydata.csv", 5)

    nt.train(250, verbose=0)

    print(f"Time for training: {time.time() - start_time} secondi")
    print(nt.make_predictions(np.array([0, 0])))

    #nt.save_model()

    #nt = nt.load_model("/Users/denisgasparollo/PycharmProjects/ransomwareNetworkTrafficDetection/src/model.pkl")

    #print(nt.make_prediction(np.array([0, 0])))


if __name__ == "__main__":
    main()