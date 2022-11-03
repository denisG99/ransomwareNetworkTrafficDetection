#Classe che permette l'estrazione delle feature che correlano maggiormente all'interno di un dataset
from dataclasses import dataclass
import pandas as pd

#TODO: da fare implemetazione algoritmi (vedere appunti)
@dataclass
class FeatureExtractor:
    __data_path : str

    def get_high_correlated_features(self, algorithm):
        df = pd.read_csv(self.__data_path, low_memory=False)
        case_dict = {} #sfrutto dizionario come sostituto allo switch-case

        return case_dict[algorithm](df)