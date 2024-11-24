from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD

data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].flatten()

y = (2 * y - 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ocsvm = OCSVM()
ocsvm.fit(X_train)

y_pred_ocsvm = ocsvm.predict(X_test)
y_pred_scores_ocsvm = ocsvm.decision_function(X_test)
y_test_pyod = (y_test + 1) // 2

balanced_acc_ocsvm = balanced_accuracy_score(y_test_pyod, y_pred_ocsvm)
roc_auc_ocsvm = roc_auc_score(y_test_pyod, y_pred_scores_ocsvm)
print(f"OCSVM Balanced Accuracy: {balanced_acc_ocsvm}")
print(f"OCSVM ROC AUC: {roc_auc_ocsvm}")

def train_deep_svdd(hidden_neurons):
    n_features = X_train.shape[1]

    if len(hidden_neurons) == 1:
        hidden_neurons = [n_features] + hidden_neurons

    deep_svdd = DeepSVDD(
            n_features=n_features,
            epochs=10,
            hidden_neurons=hidden_neurons,
            contamination=0.1
    )

    deep_svdd.fit(X_train)

    y_pred_svdd = deep_svdd.predict(X_test)
    y_pred_scores_svdd = deep_svdd.decision_function(X_test)

    balanced_acc_svdd = balanced_accuracy_score((y_test + 1) // 2, y_pred_svdd)
    roc_auc_svdd = roc_auc_score((y_test + 1) // 2, y_pred_scores_svdd)

    print(f"DeepSVDD Architecture: {hidden_neurons}")
    print(f"Balanced Accuracy: {balanced_acc_svdd}")
    print(f"ROC AUC: {roc_auc_svdd}")
    print("-" * 50)

architectures = [
    [64],
    [64, 32],
    [128, 64, 32]
]

for arch in architectures:
    train_deep_svdd(arch)