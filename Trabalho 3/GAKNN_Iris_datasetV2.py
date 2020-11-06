from sklearn.neighbors import KNeighborsClassifier
from geneticalgorithm import geneticalgorithm as ga
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import contingency_matrix

import numpy as np

def IrisKNN(K):
    
    #Implementa o Algoritmo KNN
    neigh = KNeighborsClassifier(n_neighbors=int(K[0]), leaf_size=int(76),
                                 p=int(K[2]) ,weights="uniform")
    neigh.fit(X_train, y_train)
    
    #Prevendo valores da porção de teste
    y_pred = neigh.predict(X_test)
    
    """
    Gera a Matriz de Contingência, que mostra os acertos e erros do arupamento,
    alem de especificar para qual cluster esses dados foram associados
    """
    contMatrix = contingency_matrix(y_pred, y_test)

    """
    Aqui estou percorrendo a Matriz de Contingência, calculando a porcentagem de 
    acerto para cada cluster e salvando o resultado no vetor clusterScores
    """
    nClusters = len(contMatrix)
    clusterScores = []
    hitPercentage = 0
    totalHits = 0
    globalScore = 0
    

    for i in range(nClusters):
    
        centr = np.argmax(contMatrix[i,:])
        centrValue = contMatrix[i, centr]
        soma = 0
    
        for j in range(nClusters):
            soma = soma + contMatrix[i,j]
            
        hitPercentage = centrValue/soma
        clusterScores.append(hitPercentage)
        totalHits = totalHits + centrValue

    """
    Mede a porcentagem total de acertos desconsiderando o nome dado aos clusters
    (grau de similaridade)
    """   

    globalScore = totalHits/len(y_pred)
    
    return -globalScore

#Carrega o iris dataset em iris 
iris = load_iris()
X = iris.data
y = iris.target

"""

"""
X0 = X[0:50]
X1 = X[50:100]
X2 = X[100:150]

y0 = y[0:50]
y1 = y[50:100]
y2 = y[100:150]

#divide o dataset, pegando amostras aleatórias, em porção de treino e teste
X0_train, X0_test, y0_train, y0_test = train_test_split( X0, y0, test_size=0.50, random_state=42)
X1_train, X1_test, y1_train, y1_test = train_test_split( X1, y1, test_size=0.50, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split( X2, y2, test_size=0.50, random_state=42)

X_train = np.concatenate((X0_train, X1_train, X2_train))
X_test = np.concatenate((X0_test, X1_test, X2_test))
y_train = np.concatenate((y0_train, y1_train, y2_train))
y_test = np.concatenate((y0_test, y1_test, y2_test))

varbound=np.array([[1,40], [1,10], [1,2]])

algorithm_param = {
                    'max_num_iteration': 30,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None
            }

model=ga(function=IrisKNN,dimension=3,variable_type='int',
         variable_boundaries=varbound, algorithm_parameters=algorithm_param)

model.run()

convergence=model.report
solution=model.output_dict


winnerKnnModel = KNeighborsClassifier(n_neighbors=int(model.best_variable[0]), 
                                      leaf_size=int(model.best_variable[1]),
                                      p=int(model.best_variable[2]),
                                      weights="uniform")

winnerKnnModel.fit(X_train, y_train)

#Prevendo valores da porção de teste
y_pred = winnerKnnModel.predict(X_test)


contMatrix = contingency_matrix(y_pred, y_test)
print(contMatrix)