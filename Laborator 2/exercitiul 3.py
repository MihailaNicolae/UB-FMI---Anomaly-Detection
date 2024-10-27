from pyod.models.knn import KNN
from pyod.models.lof import LOF
import matplotlib.pyplot as plt
import sklearn
import pandas as pd

model_knn = KNN(contamination=0.07)
model_lof = LOF(contamination=0.07)

X, _ = sklearn.datasets.make_blobs(n_samples=200, n_features=2, center_box=(-10,-10), cluster_std= 2, centers= 1)
X2, _ = sklearn.datasets.make_blobs(n_samples=100, n_features=2, center_box=(10,10), cluster_std= 6, centers= 1)

df_combined = pd.DataFrame(X)
df_combined = df_combined._append(pd.DataFrame(X2), ignore_index=True)
combined_data = df_combined.values

model_knn.fit(combined_data)
model_lof.fit(combined_data)
rezultate_knn = model_knn.predict(combined_data)
rezultate_lof = model_lof.predict(combined_data)

knn_normale = combined_data[rezultate_knn == 0]
knn_anomalii = combined_data[rezultate_knn == 1]
lof_normale = combined_data[rezultate_lof == 0]
lof_anomalii = combined_data[rezultate_lof == 1]

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].set_title('KNN - Results')
axs[0].scatter(knn_normale[:, 0], knn_normale[:, 1], color='blue', label='Normal data')
axs[0].scatter(knn_anomalii[:, 0], knn_anomalii[:, 1], color='red', label='Anomalies')
axs[0].set_xlabel('X axis')
axs[0].set_ylabel('Y axis')
axs[0].legend()

axs[1].set_title('LOF - Results')
axs[1].scatter(lof_normale[:, 0], lof_normale[:, 1], color='blue', label='Normal data')
axs[1].scatter(lof_anomalii[:, 0], lof_anomalii[:, 1], color='red', label='Anomalies')
axs[1].set_xlabel('X axis')
axs[1].set_ylabel('Y axis')
axs[1].legend()

plt.tight_layout()
plt.show()