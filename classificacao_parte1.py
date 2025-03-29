
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 1. Obtenha os dados das propriedades de sementes de três variedades diferentes de trigo. Analise os argumentos fornecidos, os tipos deles e as classes definidas.
seeds = fetch_openml(name='seeds')

x = seeds.data
y = seeds.target

nr_instancias = len(x)
print("Número de instâncias: ", nr_instancias)

nr_atributos = len(seeds.feature_names)
print("Número de atributos: ", nr_atributos)

print("Classes definidas: ", y.unique().tolist())

attrs = seeds.feature_names
df = pd.DataFrame(x, columns=attrs)

# 2. Pré-processamento de dados
# Verifique se todas as instâncias possuem valores, e, se for o caso, remova instâncias com valores ausentes.
valor_ausente = df.isnull().sum().sum()

if valor_ausente > 0:
   df = df.dropna()
else:
  print("Não há nenhuma instância com valor ausente.")

# Gere uma matriz de correlação entre os atributos e elimine os atributos com alta correlação.
corrs = df.corr()
plt.imshow(corrs, cmap='bwr', vmin=-1, vmax=1)
plt.title("Matriz de correlação")
plt.colorbar()
plt.show()

# Filtrando os atributos de alta correlação
ii = np.abs(corrs) >= 0.8

atributos_pouco_correlacionados = [attrs[0]]
for i in range(1,len(attrs)):
  ja_correlacionado = any(ii.values[i,0:i-1])
  if not ja_correlacionado:
    atributos_pouco_correlacionados.append(attrs[i])

print("Atributos pouco correlacionados:", atributos_pouco_correlacionados)

# Use a normalização nos atributos restantes
df_novo = df[atributos_pouco_correlacionados]

xnovo = df_novo.values[:,:-1] # remove a última coluna que é a classe
xnorm = StandardScaler().fit_transform(xnovo)

df_novo
