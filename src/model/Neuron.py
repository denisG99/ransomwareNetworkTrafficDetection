from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class Neuron:
    #TODO: realizzare dizionario contenente varie funzioni d'attivazione per fornirli maggior flessibilità

    #activation function
    def af(self, x: float) -> float: #TODO: da cambiare
        return 1/(1 + np.exp(-x))

    #weighted sum
    def weighted_sum(self, weights:np.ndarray, bias: float, x:np.ndarray) -> float:
        return np.dot(x, weights) + bias

#-----------------------------------------------------------------------------------------------------------------------

def main() -> float:
    weights: np.ndarray[np.float] = np.ones(3)
    bias: float = 0
    x: np.ndarray[np.float] = np.ones(3)
    neuron = Neuron()

    print(weights)
    print(x)

    return neuron.af(neuron.weighted_sum(weights, bias, x))

if __name__ == "__main__":
    print(main())