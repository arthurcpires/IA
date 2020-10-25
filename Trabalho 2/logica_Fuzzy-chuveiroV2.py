import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# Definindo espaço de entrada
vazao = ctrl.Antecedent(np.arange(0, 101, 1),'vazao') 
temperatura = ctrl.Antecedent(np.arange(10, 51, 1),'temperatura')
# Denifinindo espaço de saída 
potencia = ctrl.Consequent(np.arange(0, 101, 1),'potencia')

# Definindo conjuntos de Vazão e Potência automaticamente
    # Vazão Baixa(VB), Média(VM) e Alta(VA)
vazao.automf(names=['VB','VMB','VMA','VA'])
    # Potência Baixa(PB), Média(PM) e Alta(PA)
potencia.automf(names=['PB','PMB','PMA','PA'])
    

# Definindo conjuntos de Temperatura
    # Temperatura Baixa(TB), Média(TM) e Alta(TA)
temperatura['TB'] = fuzz.trapmf(temperatura.universe, [0, 10,20,25 ])
temperatura['TMB'] = fuzz.trimf(temperatura.universe, [20, 28, 35])
temperatura['TMA'] = fuzz.trimf(temperatura.universe, [25, 33, 40])
temperatura['TA'] = fuzz.trapmf(temperatura.universe, [35, 40, 50,50])


#Visualização dos conjuntos
vazao.view()
temperatura.view()
potencia.view() 


# Definindo regras             
regra1=  ctrl.Rule(temperatura['TB'] & (vazao['VB']|vazao['VMB']),
                   potencia['PB'])

regra2 = ctrl.Rule((temperatura['TB']   & (vazao['VMA']|vazao['VA']))
                   |(temperatura['TMB'] & ~vazao['VA'])
                   |(temperatura['TMA'] &  vazao['VB']),
                   potencia['PMB'])

regra3=  ctrl.Rule((temperatura['TMB']  &  vazao['VA'])
                   |(temperatura['TMA'] & ~vazao['VB'])
                   |(temperatura['TA']  & (vazao['VB'] |vazao['VMB'])),
                   potencia['PMA'])

regra4=  ctrl.Rule((temperatura['TA']  &(vazao['VMA']|vazao['VA'])),
                   potencia['PA'])
                   
regras=[regra1,regra2,regra3,regra4]
# Inserindo as regras
fuzzyPotencia_ctrl = ctrl.ControlSystem(regras) 
fuzzyPotencia = ctrl.ControlSystemSimulation(fuzzyPotencia_ctrl) 
#Inserindo valores para o cálculo
fuzzyPotencia.input['vazao'] = 50 
fuzzyPotencia.input['temperatura'] = 30
fuzzyPotencia.compute()

potencia.view(sim=fuzzyPotencia)
resultado=fuzzyPotencia.output['potencia']
print(resultado)