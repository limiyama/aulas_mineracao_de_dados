from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1.
seeds = fetch_openml(name='seeds')
x = seeds.data
y = seeds.target
attrs = seeds.feature_names

df = pd.DataFrame(x, columns=attrs)

# 2.
valor_ausente = df.isnull().sum().sum()

if valor_ausente > 0:
   df = df.dropna()
else:
  print("Não há nenhuma instância com valor ausente.")

corrs = df.corr()

ii = np.abs(corrs) >= 0.8
atributos_pouco_correlacionados = [attrs[0]]
for i in range(1,len(attrs)):
  ja_correlacionado = any(ii.values[i,0:i-1])
  if not ja_correlacionado:
    atributos_pouco_correlacionados.append(attrs[i])

df_novo = df[atributos_pouco_correlacionados]
xnovo = df_novo.values[:,:-1]
xnorm = StandardScaler().fit_transform(xnovo)

# 3.Separe 200 instâncias aleatoriamente como os dados de treino e 10 como dados de teste (usando a função train_test_split do módulo sklearn.model_selection).
X_train, X_test, y_train, y_test = train_test_split(xnorm, y, test_size=10, train_size=200)

for i in range(len(X_test)):
    print(f"Instância {i+1}: Atributos {X_test[i]} -> Classe {y_test.iloc[i]}")

# 4.  Usando o conjunto de dados de treino, realize a aprendizagem para três tipos de classificação: o método de árvore de decisão, o método Bayesiano e o
# método de vetores SVM (módulos tree, naive_bayes e svm do pacote sklearn).
tree = DecisionTreeClassifier()
bayes = GaussianNB()
svm = SVC(probability=True)

tree.fit(X_train, y_train)
bayes.fit(X_train, y_train)
svm.fit(X_train, y_train)

# 5.  Usando os três métodos, realize a classificação do conjunto de 10 instâncias de teste.
pred_tree = tree.predict(X_test)
pred_bayes = bayes.predict(X_test)
pred_svm = svm.predict(X_test)

# 6. Verifique a acurácia de classificação para todos os métodos utilizados, comparando com as classes fornecidas junto com os dados.
#A acurácia é medida como o número de instâncias com classes corretas dividido pelo número de instâncias de teste.
acc_tree = accuracy_score(y_test, pred_tree)
acc_bayes = accuracy_score(y_test, pred_bayes)
acc_svm = accuracy_score(y_test, pred_svm)

print("\nResultados das classificações das instâncias")

for i in range(len(X_test)):
    print(f"\nINSTÂNCIA {i+1}")
    print(f"Classe real: {y_test.iloc[i]}")
    print(f"Previsão Árvore de Decisão: {pred_tree[i]}")
    print(f"Previsão Naive Bayes: {pred_bayes[i]}")
    print(f"Previsão SVM: {pred_svm[i]}")

print(f"Acurácia - Árvore de Decisão: {acc_tree:.2f}")
print(f"Acurácia - Naive Bayes: {acc_bayes:.2f}")
print(f"Acurácia - SVM: {acc_svm:.2f}")
