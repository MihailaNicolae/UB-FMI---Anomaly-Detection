from pyod.utils.data import generate_data
from pyod.models import ocsvm
from pyod.models import deep_svdd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1)
X_train, X_test, y_train, y_test = generate_data(n_train=300, n_test=200, n_features=3, contamination=0.15)
# 2)
model_ocsvm_linear = ocsvm.OCSVM(kernel= 'linear', contamination=0.15)
model_ocsvm_linear.fit(X_train, y_train)
y_pred_test_linear = model_ocsvm_linear.predict(X_test)
y_scores_linear = model_ocsvm_linear.decision_function(X_test)
ba_linear = balanced_accuracy_score(y_test, y_pred_test_linear)
roc_linear = roc_auc_score(y_test, y_scores_linear)
print(ba_linear)
print(roc_linear)
# 3)
fig = plt.figure(figsize=(16, 16))

# Train - Ground Truth Linear
X_train_linear_normal_real = X_train[y_train == 0]
X_train_linear_anomalii_real = X_train[y_train == 1]

ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(
    X_train_linear_normal_real[:, 0],
    X_train_linear_normal_real[:, 1],
    X_train_linear_normal_real[:, 2],
    color='blue', label='Normal data'
)
ax1.scatter(
    X_train_linear_anomalii_real[:, 0],
    X_train_linear_anomalii_real[:, 1],
    X_train_linear_anomalii_real[:, 2],
    color='red', label='Anomalies'
)

ax1.set_title('Train - Ground Truth Linear')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')

ax1.legend()

#Train - Prediction Linear
y_pred_train_linear = model_ocsvm_linear.predict(X_train)
X_train_linear_normal_pred = X_train[y_pred_train_linear == 0]
X_train_linear_anomalii_pred = X_train[y_pred_train_linear == 1]

ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(
    X_train_linear_normal_pred[:, 0],
    X_train_linear_normal_pred[:, 1],
    X_train_linear_normal_pred[:, 2],
    color='blue', label='Normal data'
)
ax2.scatter(
    X_train_linear_anomalii_pred[:, 0],
    X_train_linear_anomalii_pred[:, 1],
    X_train_linear_anomalii_pred[:, 2],
    color='red', label='Anomalies'
)

ax2.set_title('Train - Ground Truth Linear')
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')

ax2.legend()

# Test - Ground Truth Linear
X_test_linear_normal_real = X_test[y_test == 0]
X_test_linear_anomalii_real = X_test[y_test == 1]

ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(
    X_test_linear_normal_real[:, 0],
    X_test_linear_normal_real[:, 1],
    X_test_linear_normal_real[:, 2],
    color='blue', label='Normal data'
)
ax3.scatter(
    X_test_linear_anomalii_real[:, 0],
    X_test_linear_anomalii_real[:, 1],
    X_test_linear_anomalii_real[:, 2],
    color='red', label='Anomalies'
)

ax3.set_title('Train - Ground Truth Linear')
ax3.set_xlabel('X axis')
ax3.set_ylabel('Y axis')
ax3.set_zlabel('Z axis')

ax3.legend()

# Test - Prediction Linear
y_pred_test_linear = model_ocsvm_linear.predict(X_test)
X_test_linear_normal_pred = X_test[y_pred_test_linear == 0]
X_test_linear_anomalii_pred = X_test[y_pred_test_linear == 1]

ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(
    X_test_linear_normal_pred[:, 0],
    X_test_linear_normal_pred[:, 1],
    X_test_linear_normal_pred[:, 2],
    color='blue', label='Normal data'
)
ax4.scatter(
    X_test_linear_anomalii_pred[:, 0],
    X_test_linear_anomalii_pred[:, 1],
    X_test_linear_anomalii_pred[:, 2],
    color='red', label='Anomalies'
)

ax4.set_title('Train - Ground Truth Linear')
ax4.set_xlabel('X axis')
ax4.set_ylabel('Y axis')
ax4.set_zlabel('Z axis')

ax4.legend()

##########################
plt.tight_layout()
plt.show()

# 4) Creste contamination rate-ul ca sa vedem mai bine diferenta, atat la RBF cat si la Linear
model_ocsvm_rbf = ocsvm.OCSVM(kernel= 'rbf', contamination=0.15)
model_ocsvm_rbf.fit(X_train, y_train)
y_pred_rbf = model_ocsvm_rbf.predict(X_test)

fig = plt.figure(figsize=(16, 16))

# Train - Ground Truth RBF
X_train_rbf_normal_real = X_train[y_train == 0]
X_train_rbf_anomalii_real = X_train[y_train == 1]

ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(
    X_train_rbf_normal_real[:, 0],
    X_train_rbf_normal_real[:, 1],
    X_train_rbf_normal_real[:, 2],
    color='blue', label='Normal data'
)
ax1.scatter(
    X_train_rbf_anomalii_real[:, 0],
    X_train_rbf_anomalii_real[:, 1],
    X_train_rbf_anomalii_real[:, 2],
    color='red', label='Anomalies'
)

ax1.set_title('Train - Ground Truth RBF')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')

ax1.legend()

#Train - Prediction RBF
y_pred_train_rbf = model_ocsvm_rbf.predict(X_train)
X_train_rbf_normal_pred = X_train[y_pred_train_rbf == 0]
X_train_rbf_anomalii_pred = X_train[y_pred_train_rbf == 1]

ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(
    X_train_rbf_normal_pred[:, 0],
    X_train_rbf_normal_pred[:, 1],
    X_train_rbf_normal_pred[:, 2],
    color='blue', label='Normal data'
)
ax2.scatter(
    X_train_rbf_anomalii_pred[:, 0],
    X_train_rbf_anomalii_pred[:, 1],
    X_train_rbf_anomalii_pred[:, 2],
    color='red', label='Anomalies'
)

ax2.set_title('Train - Ground Truth RBF')
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')

ax2.legend()

# Test - Ground Truth RBF
X_test_rbf_normal_real = X_test[y_test == 0]
X_test_rbf_anomalii_real = X_test[y_test == 1]

ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(
    X_test_rbf_normal_real[:, 0],
    X_test_rbf_normal_real[:, 1],
    X_test_rbf_normal_real[:, 2],
    color='blue', label='Normal data'
)
ax3.scatter(
    X_test_rbf_anomalii_real[:, 0],
    X_test_rbf_anomalii_real[:, 1],
    X_test_rbf_anomalii_real[:, 2],
    color='red', label='Anomalies'
)

ax3.set_title('Train - Ground Truth RBF')
ax3.set_xlabel('X axis')
ax3.set_ylabel('Y axis')
ax3.set_zlabel('Z axis')

ax3.legend()

# Test - Prediction RBF
y_pred_test_rbf = model_ocsvm_rbf.predict(X_test)
X_test_rbf_normal_pred = X_test[y_pred_test_rbf == 0]
X_test_rbf_anomalii_pred = X_test[y_pred_test_rbf == 1]

ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(
    X_test_rbf_normal_pred[:, 0],
    X_test_rbf_normal_pred[:, 1],
    X_test_rbf_normal_pred[:, 2],
    color='blue', label='Normal data'
)
ax4.scatter(
    X_test_rbf_anomalii_pred[:, 0],
    X_test_rbf_anomalii_pred[:, 1],
    X_test_rbf_anomalii_pred[:, 2],
    color='red', label='Anomalies'
)

ax4.set_title('Train - Ground Truth RBF')
ax4.set_xlabel('X axis')
ax4.set_ylabel('Y axis')
ax4.set_zlabel('Z axis')

ax4.legend()
##########################
plt.tight_layout()
plt.show()

# 5)
model_ocsvm_deep = deep_svdd.DeepSVDD(n_features= 3, contamination=0.15)
model_ocsvm_deep.fit(X_train, y_train)

y_pred_test_deep = model_ocsvm_deep.predict(X_test)
y_scores_deep = model_ocsvm_deep.decision_function(X_test)
ba_deep = balanced_accuracy_score(y_test, y_pred_test_deep)
roc_deep = roc_auc_score(y_test, y_scores_deep)
print(ba_deep)
print(roc_deep)

fig = plt.figure(figsize=(16, 16))

# Train - Ground Truth Deep
X_train_deep_normal_real = X_train[y_train == 0]
X_train_deep_anomalii_real = X_train[y_train == 1]

ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(
    X_train_deep_normal_real[:, 0],
    X_train_deep_normal_real[:, 1],
    X_train_deep_normal_real[:, 2],
    color='blue', label='Normal data'
)
ax1.scatter(
    X_train_deep_anomalii_real[:, 0],
    X_train_deep_anomalii_real[:, 1],
    X_train_deep_anomalii_real[:, 2],
    color='red', label='Anomalies'
)

ax1.set_title('Train - Ground Truth Deep')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')

ax1.legend()

#Train - Prediction Deep
y_pred_train_deep = model_ocsvm_deep.predict(X_train)
X_train_deep_normal_pred = X_train[y_pred_train_deep == 0]
X_train_deep_anomalii_pred = X_train[y_pred_train_deep == 1]

ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(
    X_train_deep_normal_pred[:, 0],
    X_train_deep_normal_pred[:, 1],
    X_train_deep_normal_pred[:, 2],
    color='blue', label='Normal data'
)
ax2.scatter(
    X_train_deep_anomalii_pred[:, 0],
    X_train_deep_anomalii_pred[:, 1],
    X_train_deep_anomalii_pred[:, 2],
    color='red', label='Anomalies'
)

ax2.set_title('Train - Ground Truth Deep')
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')

ax2.legend()

# Test - Ground Truth Deep
X_test_deep_normal_real = X_test[y_test == 0]
X_test_deep_anomalii_real = X_test[y_test == 1]

ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(
    X_test_deep_normal_real[:, 0],
    X_test_deep_normal_real[:, 1],
    X_test_deep_normal_real[:, 2],
    color='blue', label='Normal data'
)
ax3.scatter(
    X_test_deep_anomalii_real[:, 0],
    X_test_deep_anomalii_real[:, 1],
    X_test_deep_anomalii_real[:, 2],
    color='red', label='Anomalies'
)

ax3.set_title('Train - Ground Truth Deep')
ax3.set_xlabel('X axis')
ax3.set_ylabel('Y axis')
ax3.set_zlabel('Z axis')

ax3.legend()

# Test - Prediction Deep
y_pred_test_deep = model_ocsvm_deep.predict(X_test)
X_test_deep_normal_pred = X_test[y_pred_test_deep == 0]
X_test_deep_anomalii_pred = X_test[y_pred_test_deep == 1]

ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(
    X_test_deep_normal_pred[:, 0],
    X_test_deep_normal_pred[:, 1],
    X_test_deep_normal_pred[:, 2],
    color='blue', label='Normal data'
)
ax4.scatter(
    X_test_deep_anomalii_pred[:, 0],
    X_test_deep_anomalii_pred[:, 1],
    X_test_deep_anomalii_pred[:, 2],
    color='red', label='Anomalies'
)

ax4.set_title('Train - Ground Truth Deep')
ax4.set_xlabel('X axis')
ax4.set_ylabel('Y axis')
ax4.set_zlabel('Z axis')

ax4.legend()

##########################
plt.tight_layout()
plt.show()