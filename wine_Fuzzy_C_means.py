# -*- coding: utf-8 -*-
"""
Created on out 05 2020

@author: 
"""

from __future__ import division, print_function
import skfuzzy as fuzz
from sklearn.datasets import load_wine
import numpy as np
import matplotlib.pyplot as plt



#Carrega o wine dataset
class data0:
    data=[]
    target=[]

data = load_wine() 

# Revome 2 amostras para que fique par o numero de amostras de cada vinho
index = [0, 59]
data0.data= np.delete(data.data, index,0)
data0.target= np.delete(data.target, index,0)

# Testes para a reduzir a dimensão dos dados 
data0.data= np.delete(data0.data, [0,2,3,4,7 ],1)


# Normalização das dimensões pela média
for i in range (len(data0.data[0,:])):
    data0.data[:,i]*=(1/np.mean(data0.data[:,i]))

ng=3 # Numero de classes

# Separação dos dados em duas partes (50%) para validação
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
acertos=[]
laux=[]
delta=0.2

# m=2
# Rotina para a variação da variavel "m"
for l in range(1,100):
    #Implementa o Algoritmo Fuzzy C-means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data1t, ncenters, 1+(delta*l), error=0.005, maxiter=1000, init=None)
    cluster_membership = np.argmax(u, axis=0)
    

    #Prevendo novos valores
       
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
        data2.transpose(), cntr, 2, error=0.005, maxiter=1000)
    
    resultado= np.argmax(u, axis=0)

    k=[0,29,64,88] # Array para identificar a separação das classes
    g0=[]
    g1=[]
    g2=[]
    
   # Verificando os acertos cada classe
    for j in range(ng):
        
        k2=k[j+1]-(k[j])
                
        g0.insert(j,
                  round((len(np.where(resultado[k[j]:k[j+1]]==0)[0]))/(k2/100),2)
                  )
        g1.insert(j,
                  round((len(np.where(resultado[k[j]:k[j+1]]==1)[0]))/(k2/100),2)
                  )
        g2.insert(j,round(100-g0[j]-g1[j],2))   
    
    acertosK_means=[max(g0), max(g1),max(g2)]
    # print("\n % de acertos em cada grupo: ",acertosK_means)
    print("\n % de acerto Total: ",round(np.mean(acertosK_means),2))
    
    acertos.insert(l-1,round(np.mean(acertosK_means),2))
    laux.insert((l-1),(l*delta))
    # print(g0,g1,g2)
    # print(resultado)

# Verificando qual parametro "m" obteve maior taxa de acerto
melhor=round(laux[np.argmax(acertos)],2),acertos[np.argmax(acertos) ]    



# Grafico de barra para a taxa de acerto em relação a "m"
plt.figure(1)
plt.suptitle('Fuzzy_C_means Alterando m')
# plt.bar(laux,acertos)
plt.plot(laux,acertos)
plt.ylabel('% de acerto')
plt.xlabel('M')
plt.grid(True)
# plt.xticks(laux)
plt.ylim((0,100))
# plt.xlim((0,11))
# show() 

print("Caso com mais acertos: ",melhor)

acertos14=acertos
laux14=laux
    