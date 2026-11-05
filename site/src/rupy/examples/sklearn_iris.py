# scikit-learn in the browser: k-nearest neighbours and logistic regression on iris.
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
print("k-NN accuracy:", round(knn.score(X_test, y_test), 3))
lr = LogisticRegression(max_iter=200).fit(X_train, y_train)
print("logistic regression accuracy:", round(lr.score(X_test, y_test), 3))
print("first predictions:", lr.predict(X_test[:8]))
