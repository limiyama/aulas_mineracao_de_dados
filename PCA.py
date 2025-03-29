# 1. Carregamento dos dados
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

breast = load_breast_cancer()

# 2. Exploração de Dados
# Verifique se os dados possuem exatamente 569 amostras com trinta características (atributos) e cada amostra tem um rótulo associado a ela.
nr_amostra = len(breast.data)
print("Número de amostras: ", nr_amostra)

qnt_caracteristicas = breast.feature_names.tolist()
nr_caracteristicas = 0
for i in qnt_caracteristicas:
  nr_caracteristicas = nr_caracteristicas + 1
  print(i)

print("Número total de características: ",nr_caracteristicas)

# Coloque os dados na estrutura DataFrame do pacote pandas e mostra o começo e fim dos dados com a descrição de características.
df = pd.DataFrame(breast.data, columns=breast.feature_names)
df['label'] = pd.Series(breast.target)
df['label'].replace(0, 'Benign', inplace=True)
df['label'].replace(1, 'Malignant', inplace=True)
df

# Padronize todos os atributos usando a função StandardScaler() do módulo sklearn.preprocessing.
x = df.values[:,:-1] # remove a última coluna que é a classe de benigno/maligno
xnorm = StandardScaler().fit_transform(x)

# Verifique se os atributos normalizados tem as médias próximas à zero e desvio padrão igual 1.
print("Médias dos atributos normalizados: ", np.mean(xnorm, axis=0))
print("Desvios padrão dos atributos normalizados: ", np.std(xnorm, axis=0))

# Use duas componentes como parâmetro e determine a quantidade de variação por componente usando o resultado armazenado em estrutura pca.explained_variance_ratio_.
pca = PCA(n_components=2)
x_pca = pca.fit_transform(xnorm)

# Qual porcentagem de variação
print("Variação por 2 componentes: ", pca.explained_variance_ratio_)

# 5. Produze um gráfico de todas as instâncias dadas no espaço de duas coordenadas correspondentes as componentes principais, marcando as
# instâncias de dois rótulos de classes com cores diferentes.
df['classe'] = breast.target
target_names = breast.target_names
targets = set(df['classe'])
colors = ['r', 'b']

for target, color in zip(targets, colors):
    indices = df['classe'] == target
    plt.scatter(x_pca[indices, 0], x_pca[indices, 1], color=color, label=target_names[int(target)])

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Gráfico de 2 Componentes')
plt.legend()

# 6. Repita a analise usando 3 componentes principais
pca = PCA(n_components=3)
xpca = pca.fit_transform(xnorm)

print("Variação por 3 componentes: ", pca.explained_variance_ratio_)
