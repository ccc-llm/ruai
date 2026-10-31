"""sklearn: classifiers on iris (logistic regression, k-NN, naive Bayes), split, metrics."""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target
print(X.shape, y[:5], iris.target_names[2], iris.feature_names[0])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
print(Xtr.shape, Xte.shape, ytr[:10], yte[:10])
sc = StandardScaler().fit(Xtr)
Xtr = sc.transform(Xtr)
Xte = sc.transform(Xte)
lr = LogisticRegression().fit(Xtr, ytr)
print(lr, lr.classes_, lr.predict(Xte), round(lr.score(Xte, yte), 6))
print(np.round(lr.coef_, 1), np.round(lr.predict_proba(Xte[:3]), 2))
binary = ytr < 1
lb = LogisticRegression(C=0.5).fit(Xtr, binary.astype("int64"))
print(lb, lb.coef_.shape, np.round(lb.coef_, 1), lb.predict(Xte[:8]), np.round(lb.predict_proba(Xte[:2]), 2))
for k in [1, 3, 5]:
    knn = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
    pred = knn.predict(Xte)
    print(knn, round(accuracy_score(yte, pred), 6), pred[:12])
print(confusion_matrix(yte, KNeighborsClassifier().fit(Xtr, ytr).predict(Xte)))
nb = GaussianNB().fit(Xtr, ytr)
print(nb, nb.predict(Xte)[:15], round(nb.score(Xte, yte), 6), np.round(nb.theta_, 3), np.round(nb.class_prior_, 4))
print(np.round(nb.predict_proba(Xte[:3]), 4))
rng = np.random.default_rng(5)
R = rng.random((60, 2))
t = R[:, 0] * 3.0 - R[:, 1]
kr = KNeighborsRegressor(n_neighbors=3).fit(R[:40], t[:40])
print(kr, np.round(kr.predict(R[40:45]), 4), round(kr.score(R[40:], t[40:]), 6))
try:
    LogisticRegression().fit(Xtr, np.zeros(len(ytr), dtype="int64"))
except ValueError as e:
    print("ValueError:", e)
