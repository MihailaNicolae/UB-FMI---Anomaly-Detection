from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

###########################   Varianta 2D   ################################################################################
"""
n_samples = 500
X1, _ = make_blobs(n_samples=n_samples, centers=1, cluster_std=1.0, n_features=2, center_box=(10, 0), random_state=42)
X2, _ = make_blobs(n_samples=n_samples, centers=1, cluster_std=1.0, n_features=2, center_box=(0, 10), random_state=42)

X_train = np.vstack((X1, X2))

model_IF = IForest(contamination=0.02)
model_IF.fit(X_train)

model_DIF = DIF(hidden_neurons= [50,600],contamination=0.02)
model_DIF.fit(X_train)

model_LODA = LODA(n_bins=1000, contamination=0.02)
model_LODA.fit(X_train)

test_samples = np.random.uniform(-10, 20, size=(1000, 2))

anomaly_scores_IF = model_IF.decision_function(test_samples)
anomaly_scores_DIF = model_DIF.decision_function(test_samples)
anomaly_scores_LODA = model_LODA.decision_function(test_samples)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Isolation Forest
scatter = axs[0].scatter(test_samples[:, 0], test_samples[:, 1], c=anomaly_scores_IF, cmap='viridis')
axs[0].set_title('Isolation Forest')
axs[0].set_xlabel('Feature 1')
axs[0].set_ylabel('Feature 2')
fig.colorbar(scatter, ax=axs[0], label='Anomaly Score')

# Deep Isolation Forest
scatter = axs[1].scatter(test_samples[:, 0], test_samples[:, 1], c=anomaly_scores_DIF, cmap='viridis')
axs[1].set_title('Deep Isolation Forest')
axs[1].set_xlabel('Feature 1')
fig.colorbar(scatter, ax=axs[1], label='Anomaly Score')

# LODA
scatter = axs[2].scatter(test_samples[:, 0], test_samples[:, 1], c=anomaly_scores_LODA, cmap='viridis')
axs[2].set_title('LODA')
axs[2].set_xlabel('Feature 1')
fig.colorbar(scatter, ax=axs[2], label='Anomaly Score')

plt.suptitle('Rezultate IF, DIF, LODA')
plt.tight_layout()
plt.show()
"""

########################    Varianta 3D      ###########################################

n_samples = 500
X1, _ = make_blobs(n_samples=n_samples, centers=1, cluster_std=1.0, n_features=3, center_box=(10, 0, 5), random_state=42)
X2, _ = make_blobs(n_samples=n_samples, centers=1, cluster_std=1.0, n_features=3, center_box=(0, 10, 5), random_state=42)

X_train = np.vstack((X1, X2))

model_IF = IForest(contamination=0.02)
model_IF.fit(X_train)

model_DIF = DIF(hidden_neurons=[50, 600], contamination=0.02)
model_DIF.fit(X_train)

model_LODA = LODA(n_bins=1000, contamination=0.02)
model_LODA.fit(X_train)

test_samples = np.random.uniform(-10, 20, size=(1000, 3))

anomaly_scores_IF = model_IF.decision_function(test_samples)
anomaly_scores_DIF = model_DIF.decision_function(test_samples)
anomaly_scores_LODA = model_LODA.decision_function(test_samples)

fig = plt.figure(figsize=(18, 6))

# Isolation Forest 3D
ax1 = fig.add_subplot(131, projection='3d')
scatter = ax1.scatter(test_samples[:, 0], test_samples[:, 1], test_samples[:, 2], c=anomaly_scores_IF, cmap='viridis')
ax1.set_title('Isolation Forest')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Feature 3')
fig.colorbar(scatter, ax=ax1, label='Anomaly Score')

# Deep Isolation Forest 3D
ax2 = fig.add_subplot(132, projection='3d')
scatter = ax2.scatter(test_samples[:, 0], test_samples[:, 1], test_samples[:, 2], c=anomaly_scores_DIF, cmap='viridis')
ax2.set_title('Deep Isolation Forest')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_zlabel('Feature 3')
fig.colorbar(scatter, ax=ax2, label='Anomaly Score')

# LODA 3D
ax3 = fig.add_subplot(133, projection='3d')
scatter = ax3.scatter(test_samples[:, 0], test_samples[:, 1], test_samples[:, 2], c=anomaly_scores_LODA, cmap='viridis')
ax3.set_title('LODA')
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.set_zlabel('Feature 3')
fig.colorbar(scatter, ax=ax3, label='Anomaly Score')

plt.suptitle('Rezultate IF, DIF, LODA 3D')
plt.tight_layout()
plt.show()