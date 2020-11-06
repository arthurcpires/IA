# -*- coding: utf-8 -*-

"""
Created on Thu Oct 29 20:44:01 2020

@author: Arthur
"""

import numpy as np
import matplotlib.pyplot as plt

from control import tf, feedback, step_response, series, parallel
from geneticalgorithm import geneticalgorithm as ga

def Controlador(X):
    
    #Atribuição dos paramentros Kp, Ki e Kd
    Kp=Kpi+X[0]/100
    Ki=Kii+X[1]/100
    Kd=Kdi+X[2]/100
    
    #Função Transferência dos termos do controlador 
    P=Kp
    I=tf(Ki,[1,0])
    D=tf([Kd,0],[0.1*Kd,1])
    #União dos blocos PID 
    C=parallel(P,I,D)
    
    # Função Transferência com o Controlador
    F=series(C,G)
    
    # Penalidade para o valor do sinal de Entrada na planta
    # ou seja penalidade para o sinal de Controle alto
    tc=feedback(C,G)    
    _, yc = step_response(tc, time)
    if max(yc)>maxcontrolador:        
       #SE1 = np.square(np.subtract(1,yc))
        ISE=tdiv+max(yc)
    else:
        # Realizando a Integral do erro quadrado 
        t1 = feedback(F, 1)
        _, y1 = step_response(t1, time)
        SE = np.square(np.subtract(1,y1))
        ISE=np.sum(SE)
        
    return (ISE)

#Paramentros da Planta
K=2
T=4
#Função Transferência da Planta
G=tf(K,[T,1])
#Criando vetor de tempo para ser o mesmo em todas as análises
tdiv=201
time = np.linspace(0, 6, tdiv)

#Fator indesejado, Sinal de controle >max 
maxcontrolador=3
Kpi=0
Kii=0
Kdi=0
faixa=5*100

# Faixa de valores para as variáveis do GA
varbound=np.array([[-Kpi,faixa-Kpi],[-Kii,faixa-Kii],[-Kdi,faixa-Kdi]])

algorithm_param = {'max_num_iteration': None,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

#Rodando o GA
model=ga(function=Controlador,dimension=3,variable_type='int',variable_boundaries=varbound,
         algorithm_parameters=algorithm_param)
model.run()

convergence=model.report
solution=model.output_dict

bKp=Kpi+solution['variable'][0]/100
bKi=Kii+solution['variable'][1]/100
bKd=Kdi+solution['variable'][2]/100
bP=bKp
bI=tf(bKi,[1,0])
bD=tf([bKd,0],[0.1*bKd,1])
#União dos blocos PID 
bC=parallel(bP,bI,bD)
    
# Função Transferência com o Controlador
bF=series(bC,G)
bt = feedback(bF, 1)
_, by = step_response(bt, time)

btc= feedback(bC, G)
_, bc = step_response(btc, time)

ts = feedback(G, 1)
_, ys = step_response(ts, time)


plt.figure(1)
plt.axhline(y=1,color='r',linestyle='--',label='Entrada')
plt.plot(time, by, label='Sistema Controlado')
plt.plot(time, ys, label='Sem  Controlador')
plt.plot(time, bc, label='Sinal de Controle')
plt.grid(True)
plt.ylim([-0.1, maxcontrolador+0.2])
plt.xlim([0, 6])
plt.xlabel('tempo [s]')
plt.ylabel('sinal ')
plt.legend()
plt.title('Resposta ao degrau \n Melhor Controlador Encontrado')


