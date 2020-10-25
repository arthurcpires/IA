# -*- coding: utf-8 -*-
"""
Created on out 05 2020

@author: 
"""
from pylab import plot, show
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

import numpy as np
 
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


#Implementa o Algoritmo K-means

kmeans = KMeans(n_clusters=3, random_state=0).fit(data1)
kmeans.labels_
kmeans.cluster_centers_

#Prevendo novos valores
resultado=kmeans.predict(data2)


# a=np.where(gabarito2==0)[0]
# k=a[len(a)]

k=int(len(data2)/ng)
g0=[]
g1=[]
g2=[]
for j in range(ng):
    g0.insert(j,
              round((len(np.where(resultado[k*j:k*(j+1)]==1)[0]))/(k/100),2)
              )
    g1.insert(j,
              round((len(np.where(resultado[k*j:k*(j+1)]==0)[0]))/(k/100),2)
              )
    g2.insert(j,round(100-g0[j]-g1[j],2))   

acertosK_means=[max(g0), max(g1),max(g2)]
print("\n % de acertos em cada grupo: ",acertosK_means)
print("\n % de acerto Total: ",round(np.mean(acertosK_means),2))




