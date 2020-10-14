# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:39:36 2020

@author: Arthur
"""

from pylab import plot, show
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats


data = load_wine() 

data0 = np.array(data.data)

# Normalização das dimensões pela média
for i in range (13):
    data0[:,i]*=(1/np.mean(data0[:,i]))

# Separando os dados pelas classes
data1=data0[0:59]
data2=data0[59:130]
data3=data0[130:]

# Plot dos dados
plt.boxplot(data1)
plt.boxplot(data2)
plt.boxplot(data3)  
plt.show()


l=len(data1[0,:])
pi=[]

# Verificando quais amostras seguem distribuição normal
for i in range(l):
    k2, p = stats.normaltest(data1[:,i])
    # print(p*100)
    if p<0.05 :
        # print('reijeita normal ',i)
        pi.append(i)
    k2, p = stats.normaltest(data2[:,i])
    if p<0.05 :
        pi.append(i)
    k2, p = stats.normaltest(data3[:,i])
    if p<0.05 :
        pi.append(i)
        
indiceA=list(set(np.arange(l))-set(pi))        
indiceK=list(set(pi))

# Realizando teste Anova nas distribuições normais
for i in range(len(indiceA)):
    j=indiceA[i]
    xa,pa=stats.f_oneway(data1[:,j], data2[:,j],data3[:,j])
    print("p anova=", pa," dado =",j)

# Realizando teste Kruskal nas distribuições normais
for i in range(len(indiceK)):
    j=indiceK[i]
    xk,pk=stats.kruskal(data1[:,1], data2[:,1],data3[:,1])
    print("p kruskal =", pk," dado =",j)


# %config InlineBackend.figure_format = 'retina'


