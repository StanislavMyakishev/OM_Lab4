import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import random
from matplotlib import pyplot as plt

X0 = []
Y0 = []
Ynorm = []
for i in range(81):
    X0.append(round(-3 + 0.1 * i, 1))
    x = X0[i]
    Y0.append(round((random.uniform(-0.2, 0.2) + x / 100 + 1), 6))
    Ynorm.append(round((x / 100 + 1), 6))

X = np.array(X0).reshape((-1, 1))
Y = np.array(Y0)
# print(X)
# print(Y)

model = LinearRegression()
model.fit(X, Y)

y_pred = model.predict(X)

results = cross_val_score(model, X, Y, cv=5)
# print(results)

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = Y[train_index], Y[test_index]
    print("X_train: ", X_train)
    print("X_test ", X_test)
    print("y_train ", y_train)
    print("y_test ", y_test)

plt.plot(X0, Ynorm, 'b--', X, Y, 'r--', X, y_pred, 'g--')
plt.axis([-4, 6, 0, 2])
plt.show()
