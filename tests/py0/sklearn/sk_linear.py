"""sklearn: LinearRegression, Ridge, preprocessing, metrics, errors."""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

rng = np.random.default_rng(3)
X = rng.random((40, 3)) * 10.0
y = X @ np.array([1.5, -2.0, 0.25]) + 4.0 + rng.random(40)
for model in [LinearRegression(), LinearRegression(fit_intercept=False)]:
    model.fit(X, y)
    print(model, np.round(model.coef_, 6), np.round(model.intercept_, 6), round(model.score(X, y), 6))
for alpha in [0.1, 1.0, 10.0]:
    r = Ridge(alpha=alpha).fit(X, y)
    print(r, np.round(r.coef_, 6), np.round(r.intercept_, 6))
p = LinearRegression().fit(X[:30], y[:30]).predict(X[30:])
print(np.round(p, 4))
print(round(mean_squared_error(y[30:], p), 6), round(mean_absolute_error(y[30:], p), 6), round(r2_score(y[30:], p), 6))
s = StandardScaler()
Z = s.fit_transform(X)
print(s, np.round(s.mean_, 6), np.round(s.var_, 6), np.round(Z.mean(axis=0), 6), np.round(Z.std(axis=0), 6))
print(np.round(s.inverse_transform(Z)[:2], 6), StandardScaler(with_mean=False).fit(X).transform(X).round(3)[0])
mm = MinMaxScaler().fit(X)
print(mm, np.round(mm.data_min_, 4), np.round(mm.data_max_, 4), np.round(mm.transform(X)[:3], 4))
print(np.round(StandardScaler().fit_transform(np.array([[1.0, 5.0], [1.0, 7.0]])), 6))
for bad in [0, 1, 2, 3]:
    try:
        if bad == 0:
            LinearRegression().predict(X)
        elif bad == 1:
            LinearRegression().fit(X, y).predict(X[:, :2])
        elif bad == 2:
            LinearRegression().fit(X, y[:5])
        else:
            StandardScaler().fit(np.array([1.0, 2.0]))
    except ValueError as e:
        print("ValueError:", e)
