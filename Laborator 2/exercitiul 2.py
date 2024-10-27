import pyod.utils.data as pyod
from pyod.models.knn import KNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = pyod.generate_data_clusters(n_train=400, n_test=200, n_clusters=2, n_features=2, contamination=0.1)
model_knn = KNN(contamination=0.1, n_neighbors=5)

fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Train - Ground Truth
X_train_normal_real = X_train[y_train == 0]
X_train_anomalii_real = X_train[y_train == 1]
axs[0, 0].set_title('Train - Ground Truth')
axs[0, 0].scatter(X_train_normal_real[:, 0], X_train_normal_real[:, 1], color='blue', label='Normal data')
axs[0, 0].scatter(X_train_anomalii_real[:, 0], X_train_anomalii_real[:, 1], color='red', label='Anomalies')
axs[0, 0].set_xlabel('X axis')
axs[0, 0].set_ylabel('Y axis')
axs[0, 0].legend()

model_knn.fit(X_train, y_train)
train_results = model_knn.predict(X_train)

# Train - Predictions
X_train_normal_pred = X_train[train_results == 0]
X_train_anomalii_pred = X_train[train_results == 1]
axs[0, 1].set_title('Train - Predictions')
axs[0, 1].scatter(X_train_normal_pred[:, 0], X_train_normal_pred[:, 1], color='blue', label='Normal data')
axs[0, 1].scatter(X_train_anomalii_pred[:, 0], X_train_anomalii_pred[:, 1], color='red', label='Anomalies')
axs[0, 1].set_xlabel('X axis')
axs[0, 1].set_ylabel('Y axis')
axs[0, 1].legend()

mat_confuzie = confusion_matrix(y_train, train_results)
tp = mat_confuzie[1][1]
tn = mat_confuzie[0][0]
fn = mat_confuzie[1][0]
fp = mat_confuzie[0][1]
tpr = tp/(tp+fn)
tnr = tn/(tn+fp)
ba = (tpr + tnr) / 2
print(f'Balanced Accuracy for train data = {ba}')

# Test - Ground Truth
X_test_normal_real = X_test[y_test == 0]
X_test_anomalii_real = X_test[y_test == 1]
axs[1, 0].set_title('Test - Ground Truth')
axs[1, 0].scatter(X_test_normal_real[:, 0], X_test_normal_real[:, 1], color='blue', label='Normal data')
axs[1, 0].scatter(X_test_anomalii_real[:, 0], X_test_anomalii_real[:, 1], color='red', label='Anomalies')
axs[1, 0].set_xlabel('X axis')
axs[1, 0].set_ylabel('Y axis')
axs[1, 0].legend()

# Test - Predictions
test_results = model_knn.predict(X_test)
X_test_normal_pred = X_test[test_results == 0]
X_test_anomalii_pred = X_test[test_results == 1]
axs[1, 1].set_title('Test - Predictions')
axs[1, 1].scatter(X_test_normal_pred[:, 0], X_test_normal_pred[:, 1], color='blue', label='Normal data')
axs[1, 1].scatter(X_test_anomalii_pred[:, 0], X_test_anomalii_pred[:, 1], color='red', label='Anomalies')
axs[1, 1].set_xlabel('X axis')
axs[1, 1].set_ylabel('Y axis')
axs[1, 1].legend()

mat_confuzie = confusion_matrix(y_test, test_results)
tp = mat_confuzie[1][1]
tn = mat_confuzie[0][0]
fn = mat_confuzie[1][0]
fp = mat_confuzie[0][1]
tpr = tp/(tp+fn)
tnr = tn/(tn+fp)
ba = (tpr + tnr) / 2
print(f'Balanced Accuracy for test data = {ba}')

plt.tight_layout()
plt.show()
