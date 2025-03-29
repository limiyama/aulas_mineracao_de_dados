# 1.Carregue os dados de atributos usando a função acima.
# Coloque-os em uma estrutura DataFrame do pacote Pandas e mostre as estatísticas dos atributos (utilize a função df.describe()).
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df.describe()

# 2. Realize o agrupamento k-Means (usando sklearn.cluster.KMeans), supondo 2 grupos. Discuta como melhorar a escolha dos pontos centrais iniciais.0
# Gerar dados sintéticos
n_samples = 178
random_state = 170
X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=(2, 0.7, 1), random_state=random_state, shuffle=False)

# K-Means com 2 clusters
kmeans = KMeans(n_clusters=2)
y2 = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y2)
plt.title("KMeans com 2 Grupos")

centros = kmeans.cluster_centers_
labels = kmeans.labels_

# Cálculo da Coesão (SSE)
coesao = sum(np.linalg.norm(X[i] - centros[labels[i]]) ** 2 for i in range(len(X)))

# Cálculo da Separação (BSS)
separacao = sum(
    np.linalg.norm(centros[i] - centros[j]) ** 2
    for i in range(len(centros)) for j in range(i + 1, len(centros))
)

print(f"Coesão: {coesao:.2f}")
print(f"Separação: {separacao:.2f}")

