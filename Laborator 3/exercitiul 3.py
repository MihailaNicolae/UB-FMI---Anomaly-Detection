from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import numpy as np

# Load data
data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].ravel()  # Flatten y to 1D for compatibility with sklearn

# Initialize lists to store the BA and ROC AUC scores for each model
ba_if, roc_auc_if = [], []
ba_dif, roc_auc_dif = [], []
ba_loda, roc_auc_loda = [], []

# Number of splits
#n_splits = 10
n_splits = 2 ####### ATENTIE! Merge cu orice n_splits, dar merge SUUUPER LENT. Las cu 2 ca sa vedeti ca merge si aveti mai sus cu 10

# Run the experiment over multiple splits
for _ in range(n_splits):
    # Split data into train and test sets (40% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize models
    model_if = IForest()
    model_dif = DIF(n_estimators=1)
    model_loda = LODA()

    # Fit and evaluate IForest
    model_if.fit(X_train)
    y_pred_if = model_if.predict(X_test)
    y_scores_if = model_if.decision_function(X_test)
    ba_if.append(balanced_accuracy_score(y_test, y_pred_if))
    roc_auc_if.append(roc_auc_score(y_test, y_scores_if))

    # Fit and evaluate Deep Isolation Forest (DIF)
    # Partea asta dintre Checkpoints este cea care dureaza EXTREM DE MULT. Nu-mi dau seama de ce
    model_dif.fit(X_train)
    #print('Checkpoint 1')
    y_pred_dif = model_dif.predict(X_test)
    y_scores_dif = model_dif.decision_function(X_test)
    ba_dif.append(balanced_accuracy_score(y_test, y_pred_dif))
    roc_auc_dif.append(roc_auc_score(y_test, y_scores_dif))
    #print('Checkpoint 2')

    # Fit and evaluate LODA
    model_loda.fit(X_train)
    y_pred_loda = model_loda.predict(X_test)
    y_scores_loda = model_loda.decision_function(X_test)
    ba_loda.append(balanced_accuracy_score(y_test, y_pred_loda))
    roc_auc_loda.append(roc_auc_score(y_test, y_scores_loda))

# Compute mean and standard deviation of BA and ROC AUC for each model
results_if = {
    "Balanced Accuracy": (np.mean(ba_if), np.std(ba_if)),
    "ROC AUC": (np.mean(roc_auc_if), np.std(roc_auc_if))
}
results_dif = {
    "Balanced Accuracy": (np.mean(ba_dif), np.std(ba_dif)),
    "ROC AUC": (np.mean(roc_auc_dif), np.std(roc_auc_dif))
}
results_loda = {
    "Balanced Accuracy": (np.mean(ba_loda), np.std(ba_loda)),
    "ROC AUC": (np.mean(roc_auc_loda), np.std(roc_auc_loda))
}

# Print results for each model
print("IForest Results:")
print(f"  Balanced Accuracy: {results_if['Balanced Accuracy'][0]:.3f} ± {results_if['Balanced Accuracy'][1]:.3f}")
print(f"  ROC AUC: {results_if['ROC AUC'][0]:.3f} ± {results_if['ROC AUC'][1]:.3f}\n")

print("Deep Isolation Forest (DIF) Results:")
print(f"  Balanced Accuracy: {results_dif['Balanced Accuracy'][0]:.3f} ± {results_dif['Balanced Accuracy'][1]:.3f}")
print(f"  ROC AUC: {results_dif['ROC AUC'][0]:.3f} ± {results_dif['ROC AUC'][1]:.3f}\n")

print("LODA Results:")
print(f"  Balanced Accuracy: {results_loda['Balanced Accuracy'][0]:.3f} ± {results_loda['Balanced Accuracy'][1]:.3f}")
print(f"  ROC AUC: {results_loda['ROC AUC'][0]:.3f} ± {results_loda['ROC AUC'][1]:.3f}\n")