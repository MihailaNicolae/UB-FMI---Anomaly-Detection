import numpy as np
import matplotlib.pyplot as plt

# Generam datele
mean = [5, 10, 2]
cov = [[3, 2, 2], [2, 10, 1], [2, 1, 2]]
data = np.random.multivariate_normal(mean, cov, 500)

data_centered = data - np.mean(data, axis=0)

# Coalculam matricea de covarianta si facem EVD
cov_matrix = np.cov(data_centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sortam valorile si vectorii proprii
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Afisam grafic explained variance
cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

plt.figure(figsize=(8, 5))
plt.bar(range(1, 4), eigenvalues / np.sum(eigenvalues), alpha=0.6, label='Individual variance')
plt.step(range(1, 4), cumulative_variance, where='mid', label='Cumulative variance', color='red')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained')
plt.title('Explained Variance')
plt.legend()
plt.grid(True)

# Adaugam in grafic cumulative variance
for i, cv in enumerate(cumulative_variance):
    plt.text(i + 1, cv, f"{cv:.2f}", ha='center', va='bottom', fontsize=9, color='black')

plt.show()

# Proiectam datele si gasim anomaliile
projected_data = np.dot(data_centered, eigenvectors)
threshold_2 = np.quantile(projected_data[:, 1], 0.9)  # trheshold a doua componenta
threshold_3 = np.quantile(projected_data[:, 2], 0.9)  # threshold a treia componenta

# Afisam anomaliile pe a doua si a treia componenta principala
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(*data.T, c=(projected_data[:, 2] > threshold_3), cmap='coolwarm')
ax1.set_title('Outliers on 3rd Principal Component')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(*data.T, c=(projected_data[:, 1] > threshold_2), cmap='coolwarm')
ax2.set_title('Outliers on 2nd Principal Component')

plt.show()

# Calculam distantele normalizate pana la centroid
centroid = np.mean(projected_data, axis=0)  # gasim centroidul
std_devs = np.std(projected_data, axis=0)   # deviata standard

normalized_distances = np.sqrt(np.sum(((projected_data - centroid) / std_devs) ** 2, axis=1))

# Stabilim pragul
threshold = np.quantile(normalized_distances, 0.9)
anomalies = normalized_distances > threshold

# Afisam grafic rezultatele
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Date normale
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', label='Normal Points', s=20)

# Anomalii
ax.scatter(data[anomalies, 0], data[anomalies, 1], data[anomalies, 2],
           c='r', label='Anomalies', s=40, edgecolor='k')

ax.set_title('Outliers Based on Normalized Distance in PCA Space')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
plt.show()