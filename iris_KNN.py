# -*- coding: utf-8 -*-
"""
Created on out 05 2020

@author: 
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np
from pylab import plot, show
 
#Carrega o iris dataset em iris 
data0 = load_iris()

ng=3


data1 = np.array(data0.data[0])
data2 = np.array([data0.data[1]])
gabarito1=np.array([data0.target[0]])
gabarito2=np.array([data0.target[1]])
for i in range(2,len(data0.data),2):
    data1=np.vstack((data1,data0.data[i]))
    data2=np.vstack((data2, data0.data[i+1]))
    gabarito1=np.vstack((gabarito1,data0.target[i]))
    gabarito2=np.vstack((gabarito2,data0.target[i+1]))


X = data1
y = np.ravel(gabarito1)
yy=data0.target


#iris.target
#Implementa o Algoritmo KNN
neigh = KNeighborsClassifier(n_neighbors=3,weights="uniform")
neigh.fit(X, y)

#Prevendo novos valores
resultado=neigh.predict(data2)

k=int(len(data2)/ng)
g0=[]
g1=[]
g2=[]
for j in range(ng):
    g0.insert(j,
              round((len(np.where(resultado[k*j:k*(j+1)]==0)[0]))/(k/100),2)
              )
    g1.insert(j,
              round((len(np.where(resultado[k*j:k*(j+1)]==1)[0]))/(k/100),2)
              )
    g2.insert(j,round(100-g0[j]-g1[j],2))   

acertosK_means=[max(g0), max(g1),max(g2)]
print("\n % de acertos em cada grupo: ",acertosK_means)
print("\n % de acerto Total: ",round(np.mean(acertosK_means),2))

