from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

def compute_anomaly_score(sample, histograms, projections):
    scores = []
    for (counts, bin_edges), proj in zip(histograms, projections):
        projected_value = sample @ proj
        # Gasim bin-ul pentru data proiectata
        bin_index = np.searchsorted(bin_edges, projected_value, side='right') - 1
        # Verificam ca indexul sa fie din intervalul valid
        if 0 <= bin_index < len(counts):
            prob = counts[bin_index]
        else:
            prob = 0
        scores.append(prob)
    # Anomaly score = media probabilitatilor din toate histogramele
    return np.mean(scores)

# Generare set date 2D
n_samples = 500
X, _ = make_blobs(n_samples=n_samples, centers=1, cluster_std=1.0, n_features=2, random_state=42)

# Reprezentam grafic setul 2D initial
plt.scatter(X[:, 0], X[:, 1])
plt.title("Generated 2D Dataset")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# Generam projection vectors
n_projections = 5
projections = []

for _ in range(n_projections):
    vec = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    unit_vec = vec / np.linalg.norm(vec)  # Normalizam
    projections.append(unit_vec)

# Proiectam datele pe vectori si facem histogramele
histograms = []
bin_counts = 100  # Nr bins
range_interval = (-150, 150)  # de aici modificam acoperirea tuturor valorilor punctelor

for proj in projections:
    # Proiectam datele pe vectori
    projected_data = X @ proj

    # Facem histogramele
    counts, bin_edges = np.histogram(projected_data, bins=bin_counts, range=range_interval, density=True)
    histograms.append((counts, bin_edges))

# Calculam anomaly score pentru punctele originale
anomaly_scores = np.array([compute_anomaly_score(sample, histograms, projections) for sample in X])

# Generam datele de test
test_samples = np.random.uniform(-3, 3, size=(n_samples, 2))

# Calculam anomaly score pentru datele de test
test_anomaly_scores = np.array([compute_anomaly_score(sample, histograms, projections) for sample in test_samples])

# Reprezentam grafic punctele
plt.scatter(test_samples[:, 0], test_samples[:, 1], c=test_anomaly_scores, cmap='viridis', s=20)
plt.colorbar(label='Anomaly Score')
plt.title("Test Dataset with Anomaly Scores")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()