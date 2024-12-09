import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from pyod.utils.utility import standardizer
from scipy.io import loadmat

# Datasetul shuttle
data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].ravel()

# Impartim datele 60-40 train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardizam datele
X_train, X_test = standardizer(X_train, X_test)

# Antrenam PCA-ul
contamination_rate = y_train.sum() / len(y_train)  # Real contamination rate
pca_model = PCA(contamination=contamination_rate, n_components=3)
pca_model.fit(X_train)

# Calculam explained variance si cumulative variance
explained_variance = pca_model.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Reprezentam grafic individual variances si cumulative variance
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, label='Individual variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative variance', color='red')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.title('Explained Variance for PCA')
plt.legend()
plt.grid(True)
plt.show()

# Calculam balanced accuracy
y_train_pred = pca_model.predict(X_train)
y_test_pred = pca_model.predict(X_test)

train_accuracy = balanced_accuracy_score(y_train, y_train_pred)
test_accuracy = balanced_accuracy_score(y_test, y_test_pred)

print(f"Balanced Accuracy (PCA) - Train: {train_accuracy:.4f}")
print(f"Balanced Accuracy (PCA) - Test: {test_accuracy:.4f}")

# Repetam pentru KPCA
# Am ales kernel linear fiindca avem multe date, iar matricea kernel e prea mare pentru laptopul meu (memorie + procesare)
kpca_model = KPCA(contamination=contamination_rate,  kernel='linear', n_components=3)
# Am ales float32 ca sa reduc povara computationala. Mi-a luat 30 de minute sa fac fit cu parametrii astia
kpca_model.fit(X_train.astype(np.float32))

# Calculam balanced accuracy pentru KPCA
y_train_pred_kpca = kpca_model.predict(X_train.astype(np.float32))
y_test_pred_kpca = kpca_model.predict(X_test.astype(np.float32))

train_accuracy_kpca = balanced_accuracy_score(y_train, y_train_pred_kpca)
test_accuracy_kpca = balanced_accuracy_score(y_test, y_test_pred_kpca)

print(f"Balanced Accuracy (KPCA) - Train: {train_accuracy_kpca:.4f}")
print(f"Balanced Accuracy (KPCA) - Test: {test_accuracy_kpca:.4f}")