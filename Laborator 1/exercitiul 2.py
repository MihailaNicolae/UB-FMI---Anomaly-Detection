#Exercice 2
import pyod.utils.data as pyod
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

X_train, X_test, y_train, y_test = pyod.generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1)
X_train_normal = X_train[y_train == 0]
X_train_anomalii = X_train[y_train == 1]

model_knn = KNN(contamination=0.1)
#model_knn = KNN(contamination=0.5)
model_knn.fit(X_train,y_train)
results = model_knn.predict(X_train)

mat_confuzie = confusion_matrix(y_train,results)
tp = mat_confuzie[1][1]
tn = mat_confuzie[0][0]
fn = mat_confuzie[1][0]
fp = mat_confuzie[0][1]

tpr = tp/(tp+fn)
tnr = tn/(tn+fp)
ba = (tpr+tnr)/2

print(ba)

scores = model_knn.predict_confidence(X_train)
#print(scores)
fpr, tpr, thresholds = roc_curve(y_train, scores)
plt.figure()
plt.plot(tpr, fpr, color='blue', lw=2, label=f'ROC curve')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()