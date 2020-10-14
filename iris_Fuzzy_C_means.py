# -*- coding: utf-8 -*-
"""
Created on out 05 2020

@author: 
"""

from __future__ import division, print_function
import skfuzzy as fuzz
from sklearn.datasets import load_iris
import numpy as np
from pylab import plot, show


colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

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


data1t = data1.transpose()
label = data0.target 
ncenters = ng


#Implementa o Algoritmo Fuzzy C-means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data1t, ncenters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)


#Prevendo novos valores


u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    data2.transpose(), cntr, 2, error=0.005, maxiter=1000)

resultado= np.argmax(u, axis=0)


k=int(len(data2)/ng)
g0=[]
g1=[]
g2=[]
for j in range(ng):
    g0.insert(j,(len(np.where(resultado[k*j:k*(j+1)]==0)[0]))/(k/100))
    g1.insert(j,(len(np.where(resultado[k*j:k*(j+1)]==1)[0]))/(k/100))
    g2.insert(j,round(100-g0[j]-g1[j],2))   

acertos=[max(g0), max(g1),max(g2)]
print("\n % de acertos em cada grupo: ",acertos)
print("\n % de acerto Total: ",np.mean(acertos))