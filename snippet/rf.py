'''
bozza algoritmo random forest: esempio classificatore RF su capture_download iris
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler #normalizzazione dei dati
from sklearn.datasets import load_iris

#1. Import del capture_download
iris = load_iris()

#2. Divido capture_download in train e test set
iris_data, iris_label = iris['data'], iris['target']

X_train, X_test, Y_train, Y_test = train_test_split(iris_data, iris_label, test_size=0.3, train_size=0.7)

#3. Normalizzazione dei dati


#4. Fase di training
'''
 * n_estimator: numero alberi foresta (default 100)
 * criterion: funzione che misura qualità della divisione ({'gini', 'entropy'} default 'gini')
 * max_depth: indica profondità massima dell'albero (default: None -> espansione avviene fino a ottenere foglie pure o 
              contengono fino a una certa quantità di campioni)
 * min_samples_split: numero minimo di sample necessari per poter effettuare uno slit, il quale può essere un intero o 
                      un float (default 2)
 * min_samples_leaf: numero minimo di sample per poter considerare un nodo una foglia (default 1)
 * min_weight_fraction_leaf: frazione pesata minima della somma dei pesi richiesta per trovarsi in una foglia (default 0.0)
 * max_features: numero di features da tenere in considerazione quando cerco lo split migliore ({int, float, auto, sqrt, log2} default auto)
 Per altro vedere documentazione: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
 '''
classfier = RandomForestClassifier(criterion='gini')
classfier.fit(X_train, Y_train)

#5. Testing classificatore
pred = classfier.predict(X_test)

#6. Valutazione dell'algoritmo
'''
 * confusion matrix
 * risultati in fase di training
 * risultati in fase di testing
'''
cm = confusion_matrix(Y_test, pred)
acc = precision_score(Y_test, pred, average='micro')
print(cm)
print(f"Accuracy: {acc}")