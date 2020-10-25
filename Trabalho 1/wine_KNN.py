# -*- coding: utf-8 -*-
"""
Created on out 05 2020

@author: 
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
import numpy as np
# from pylab import plot, show
import matplotlib.pyplot as plt

 
#Carrega o wine dataset em wine 
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

ng=3 # Numero de classesv

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


X = data1
y = np.ravel(gabarito1)
yy=data0.target

acertos=[]
laux=[]



#  lmax 89
# Rotina para a variação da variavel "k"
for l in range(1,89):
    #Implementa o Algoritmo KNN
    neigh = KNeighborsClassifier(n_neighbors=l,weights="uniform")
    neigh.fit(X, y)
    
    #Prevendo novos valores
    resultado=neigh.predict(data2)
    
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
    print("\n % de acertos em cada grupo: ",acertosK_means)
    print("\n % de acerto Total: ",round(np.mean(acertosK_means),2))
    
    # l=0
    acertos.insert(l-1,round(np.mean(acertosK_means),2))
    laux.insert(l-1,l)

# Verificando qual parametro "k" obteve maior taxa de acerto
melhor=laux[np.argmax(acertos)],acertos[np.argmax(acertos) ]
    
# Grafico de barra para a taxa de acerto em relação a "k"
plt.figure(1)
plt.suptitle('KNN Alterando Vizinhança \n Data Normalizada e Reduzida')
# plt.bar(laux,acertos)
plt.plot(laux,acertos)
plt.ylabel('% de acerto')
plt.xlabel('N')
plt.grid(True)
# plt.xticks(laux)
plt.ylim((0,100))

print("Caso com mais acertos: ",melhor)

# acertos4=melhor
# laux4=laux
    
    
    
