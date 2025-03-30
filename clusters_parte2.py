import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import homogeneity_score, completeness_score, silhouette_score
import numpy as np


wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# 4. Realize um agrupamento por densidade (sklearn.cluster.DBSCAN). Teste como o agrupamento depende dos parâmetros eps e min_samples.
db1 = DBSCAN(eps=1, min_samples=5).fit(X)
ydb1 = db1.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=ydb1)
plt.title("eps = 1 e min_samples = 5")
plt.show()

db2 = DBSCAN(eps=1, min_samples=10).fit(X)
ydb2 = db2.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=ydb2)
plt.title("eps = 1 e min_samples = 10")
plt.show()

db3 = DBSCAN(eps=2, min_samples=5).fit(X)
ydb3 = db3.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=ydb3)
plt.title("eps = 2 e min_samples = 5")
plt.show()

n_samples = 178
random_state = 170
X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=(2, 0.7, 1), random_state=random_state, shuffle=False)

y2 = KMeans(n_clusters=2).fit_predict(X)
y3 = KMeans(n_clusters=3).fit_predict(X)
db = DBSCAN(eps=1, min_samples=5).fit(X)
ydb = db.fit_predict(X)

model = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X)
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='average')
model = model.fit(X)

yh = model.labels_

# 5. Para cada um desses agrupamentos, realize a avaliação utilizando as medidas de homogeneidade, integridade e coeficiente de Silhouette.
print("Avaliação agrupamento KMeans - 2 grupos")
print("Homogeneidade: %0.3f" % homogeneity_score(y, y2))
print("Integridade: %0.3f" % completeness_score(y, y2))
print("Coeficiente de Silhouette: %0.3f" % silhouette_score(X, y2))

print("\n Avaliação agrupamento KMeans - 3 grupos")
print("Homogeneidade: %0.3f" % homogeneity_score(y, y3))
print("Integridade: %0.3f" % completeness_score(y, y3))
print("Coeficiente de Silhouette: %0.3f" % silhouette_score(X, y3))

print("\n Avaliação agrupamento hierárquico")
print("Homogeneidade: %0.3f" % homogeneity_score(y, yh))
print("Integridade: %0.3f" % completeness_score(y, yh))

print("\n Avaliação agrupamento DBSCAN")
print("Homogeneidade: %0.3f" % homogeneity_score(y, ydb))
print("Integridade: %0.3f" % completeness_score(y, ydb))
print("Coeficiente de Silhouette: %0.3f" % silhouette_score(X, ydb))
