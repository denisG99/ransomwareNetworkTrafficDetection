from dataclasses import dataclass, field

from Node import Node
from NodeType import NodeType
from Classification import Classification

import pandas as pd

@dataclass
class NeuralTree:
    """
    Costruttore:
        NeuralTree(__data_path,
                   __wait_epochs)
        """
    __root : Node = field(init=False)
    __data_path : str
    #TREE STATISTICS
    """
    __depth: int = 0 #depth of tree
    __num_nodes : int = 0 #count the number of node
    __num_splitted : int = 0 #count the number of splittin node
    __num_leaf : int = 0 #count the number of leaf node
    """
    #__toler : float = 0.0 #tollerance that indicate the end of the branch training. Will be use in convergence test
    __wait_epochs : int = 0 #wait epochs that wait before splitting node if the boundaries don't get any improvement

    def __post_init__(self):
        data = pd.read_csv(self.__data_path, low_memory=False).to_numpy()

        self.__root = Node(data, data.shape[1] - 1)
        #self.__num_nodes += 1

    def train(self, epochs: int, verbose: bool = False) -> None:
        for epoch in range(epochs):
            if verbose:
                print(f"-- Epoch {epoch + 1}")

            model = self.__root.train()

            if verbose:
                print(f"\tWeights: {model.coef_}")
                print(f"\tBias: {model.intercept_}")
                print(f"\n")

#-----------------------------------------------------------------------------------------------------------------------

def main() -> None:
    nt = NeuralTree("./toydata.csv")

    nt.train(1)

    print(nt)

if __name__ == "__main__":
    main()