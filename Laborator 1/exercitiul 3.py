#Exercice 3
import pyod.utils.data as pyod
from sklearn.metrics import confusion_matrix
import numpy as np

X_train, X_test, y_train, y_test = pyod.generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1)

miu = np.mean(X_train)
sigma = np.std(X_train)
z_scores = (X_train - miu)/sigma

threshold = np.quantile(np.abs(z_scores), 1 - 0.1)
anomaly_predictions = np.abs(z_scores) > threshold

y_true = np.zeros_like(X_train)
y_true[np.abs(z_scores) > threshold] = 1

mat_confuzie = confusion_matrix(y_train,y_true)
tp = mat_confuzie[1][1]
tn = mat_confuzie[0][0]
fn = mat_confuzie[1][0]
fp = mat_confuzie[0][1]

tpr = tp/(tp+fn)
tnr = tn/(tn+fp)
ba = (tpr+tnr)/2

print(ba)