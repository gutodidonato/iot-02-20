# Importar as bibliotecas necessárias

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def limparTela(x):
    '''
    Pula x linhas
    '''
    for n in range(x):
        print("\n")

diamond = pd.read_csv('datasets\diamonds.csv', sep=",")
print(f"Tamanho de : {diamond.shape}")
diamond.head(10)
diamond.tail(10)

#linhas são atributos 
#colunas são registros

print(diamond.columns)
print(diamond.info())

#Dtype de objeto são classes == dados categóricos

print(diamond.describe())
print(diamond.describe(include='object'))

preco = diamond['price']
limparTela(1)
limparTela(2)

preco_tamanho = diamond[['price', 'x', 'y']]
print(preco_tamanho[:10])

limparTela(1)
print("media = " + str(preco.mean().round(4)))
print("mediana = " + str(preco.median().round(4)))
print("valor maximo = " + str(preco[max].round(4)))
print("desvio padrão é = " + str(preco.std().round(4)))
limparTela(1)

diamond.drop("Unnamed: 0", axis=1, inplace=True) 
print(diamond.info())  
print(diamond.head())  

vetor_numpy = preco.values
vetor_corte = diamond['cut'].values

le = LabelEncoder()
diamond['cut'] = le.fit_transform(diamond['cut'])

print(diamond.head())


#X maiusculo é dado de entrada/feature/ variaveis independentes
#y minusculo é o target/dados de saida
#iloc > [linhas , colunas] ---> por indice
#loc > [linhas, colunas] ----> por rotulo

tamanhos = diamond.iloc[:, 7:]
print(tamanhos[:5])
y = preco