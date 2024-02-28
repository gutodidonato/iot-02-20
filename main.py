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
y = diamond['cut']
print(diamond.head())

#label encoder gera dados tabelares para o cut, isso é valores numéricos
#X maiusculo é dado de entrada/feature/ variaveis independentes
#y minusculo é o target/dados de saida
#iloc > [linhas , colunas] ---> por indice
#loc > [linhas, colunas] ----> por rotulo

tamanhos = diamond.iloc[:, 7:]
print(tamanhos[:5])
print(y[:5])


#ele já embaralha a amostra pelo train_text_split
x_train, x_test, y_train, y_test = train_text_split(tamanhos, y, test_size=0.30, random_state=42)

print(x_train)

modelo_KNN = KNeighborsClassifier(n_neighbors=5)
modelo_KNN.fit(x_train, y_train)

#superviosionada pq você consegue verificar a acertividade
y_predict_KNN = modelo_KNN.predict(x_test)
y_predict_KNNP

acuracia = accuracy_score(y_test, y_predict_KNN)

'''
=======================
    DECISION TREE
=======================
'''

modelo_tree = DecisionTreeClassifier()
modelo_tree.fit(x_train, x_test)

y_predict_tree = modelo_tree.predict(x_test)
acuracia2 = accuracy_score(y_test, y_predict_tree)
