#Exercice 1
import pyod.utils.data as pyod
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = pyod.generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1)
X_train_normal = X_train[y_train == 0]
X_train_anomalii = X_train[y_train == 1]

plt.scatter(X_train_normal[:, 0], X_train_normal[:, 1], color='blue', label='Normal data')
plt.scatter(X_train_anomalii[:, 0], X_train_anomalii[:, 1], color='red', label='Anomalies')

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.show()
