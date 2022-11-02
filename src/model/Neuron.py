from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True, repr=False)
class Neuron:
    #activation function
    def af(self, x: float) -> int:
        return 1 if x > 0 else 0

    #weighted sum
    def weighted_sum(self, weights:np.ndarray[np.float], bias: float, x:np.ndarray[np.float]) -> float:
        return np.dot(x, weights) + bias


def main() -> int:
    weights: np.ndarray[np.float] = np.ones(3)
    bias: float = 0
    x: np.ndarray[np.float] = np.ones(3)
    neuron = Neuron()

    print(weights)
    print(x)

    return neuron.af(neuron.weighted_sum(weights, bias, x))

if __name__ == "__main__":
    print(main())