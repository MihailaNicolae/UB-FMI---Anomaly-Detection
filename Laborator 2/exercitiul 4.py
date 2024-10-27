import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from pyod.utils.utility import standardizer
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.combination import average
from scipy.stats import mode

data = scipy.io.loadmat('cardio.mat')
X = data['X']
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_neighbors_range = range(30, 121, 10)
train_scores = []
test_scores = []

for n_neighbors in n_neighbors_range:
    model = KNN(n_neighbors=n_neighbors)
    model.fit(X_train)

    train_scores.append(model.decision_scores_)
    test_scores.append(model.decision_function(X_test))

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    ba_train = balanced_accuracy_score(y_train, y_train_pred)
    ba_test = balanced_accuracy_score(y_test, y_test_pred)
    print(f"n_neighbors={n_neighbors}, Train BA: {ba_train:.4f}, Test BA: {ba_test:.4f}")

train_scores = np.array(train_scores)
test_scores = np.array(test_scores)

train_scores_norm = []
test_scores_norm = []

for train_score, test_score in zip(train_scores, test_scores):
    train_score_norm, test_score_norm = standardizer(train_score.reshape(-1, 1), test_score.reshape(-1, 1))
    train_scores_norm.append(train_score_norm.ravel())
    test_scores_norm.append(test_score_norm.ravel())

train_scores_norm = np.array(train_scores_norm)
test_scores_norm = np.array(test_scores_norm)

avg_train_score = average(train_scores_norm)
avg_test_score = average(test_scores_norm)

contamination = 0.176
threshold_avg = np.quantile(avg_train_score, 1 - contamination)

predictions_avg = []
predictions_max = []

for test_score in test_scores_norm:
    pred_avg = (test_score >= threshold_avg).astype(int)

    predictions_avg.append(pred_avg)

predictions_avg = np.array(predictions_avg)

y_pred_avg = mode(predictions_avg, axis=0)[0].flatten()

ba_avg = balanced_accuracy_score(y_test, y_pred_avg)
print(f"Final Balanced Accuracy (Average Strategy): {ba_avg:.4f}")