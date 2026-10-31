"""sklearn: PCA and KMeans (from given centers)."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]])
k = KMeans(n_clusters=2, init=X[:2], n_init=1).fit(X)
print(k)
print(k.labels_, k.cluster_centers_, np.round(k.inertia_, 6), k.n_iter_, k.predict(np.array([[0.0, 0.0], [10.0, 10.0]])))
iris = load_iris().data
k3 = KMeans(n_clusters=3, init=iris[[0, 50, 100]], n_init=1).fit(iris)
print([int((k3.labels_ == c).sum()) for c in range(3)], np.round(k3.cluster_centers_, 4), round(float(k3.inertia_), 4), k3.n_iter_)
p = PCA(n_components=2).fit(X)
print(p, np.round(p.components_, 6), np.round(p.explained_variance_ratio_, 6), np.round(p.singular_values_, 6), np.round(p.mean_, 6))
print(np.round(p.transform(X), 4))
pi = PCA().fit(iris)
print(pi, np.round(pi.explained_variance_, 5), np.round(pi.explained_variance_ratio_, 5), np.round(pi.components_, 4))
print(np.round(PCA(n_components=2).fit_transform(iris)[:4], 4))
try:
    PCA(n_components=5).fit(X)
except ValueError as e:
    print("ValueError")
