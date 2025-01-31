import numpy as np
import random
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score

class IsolationTreeNode:
    def __init__(self, split_feature=None, split_value=None, left=None, right=None, size=0, is_leaf=False):
        self.split_feature = split_feature  # Feature index to split on
        self.split_value = split_value      # Value of the split
        self.left = left                    # Left subtree
        self.right = right                  # Right subtree
        self.size = size                    # Number of samples at this node
        self.is_leaf = is_leaf              # Whether this node is a leaf

def compute_pgif_split(data, feature_index, k=2):
    unique_values = np.sort(np.unique(data[:, feature_index])) # Extract the unique sorted values of the feature

    if len(unique_values) <= 1:
        return unique_values[0]  # No possible split

    segment_lengths = np.diff(unique_values)  # Differences between consecutive values

    # Compute probabilities for splits (prob ~ segment_length^k)
    probabilities = segment_lengths ** k
    probabilities /= np.sum(probabilities)  # Normalize to sum to 1

    # Choose a segment with probability proportional to its length^k
    chosen_index = np.random.choice(len(segment_lengths), p=probabilities)

    # Pick a split value within the chosen segment
    split_value = (unique_values[chosen_index] + unique_values[chosen_index + 1]) / 2
    return split_value


def build_pgif_tree(data, current_depth=0, max_depth=10):
    num_samples, num_features = data.shape

    # Stopping conditions: If the node has only one data point or max depth is reached, create a leaf node
    if num_samples <= 1 or current_depth >= max_depth:
        return IsolationTreeNode(is_leaf=True, size=num_samples)

    # Randomly select a feature to split on
    feature_index = random.randint(0, num_features - 1)

    # Compute a probabilistic split point using PGIF
    split_value = compute_pgif_split(data, feature_index)

    # Partition the data based on the split value
    left_mask = data[:, feature_index] < split_value
    right_mask = ~left_mask

    # Recursively build left and right subtrees
    left_subtree = build_pgif_tree(data[left_mask], current_depth + 1, max_depth)
    right_subtree = build_pgif_tree(data[right_mask], current_depth + 1, max_depth)

    # Create the current node
    return IsolationTreeNode(
        split_feature=feature_index,
        split_value=split_value,
        left=left_subtree,
        right=right_subtree,
        size=num_samples
    )

class PGIForest:
    def __init__(self, num_trees=100, max_samples=256, max_depth=10, k=2):
        self.num_trees = num_trees
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.k = k
        self.trees = []

    def fit(self, X):
        self.trees = []
        num_samples = X.shape[0]

        for _ in range(self.num_trees):
            # Sample data points randomly for each tree
            sample_indices = np.random.choice(num_samples, self.max_samples, replace=False)
            sample_data = X[sample_indices]

            # Build and store the PGIF tree
            tree = build_pgif_tree(sample_data, max_depth=self.max_depth)
            self.trees.append(tree)

    def path_length(self, x, tree, current_depth=0):
        if tree.is_leaf:
            return current_depth + np.log2(tree.size + 1)

        if x[tree.split_feature] < tree.split_value:
            return self.path_length(x, tree.left, current_depth + 1)
        else:
            return self.path_length(x, tree.right, current_depth + 1)

    def anomaly_score(self, X):
        avg_path_lengths = np.mean([[self.path_length(x, tree) for tree in self.trees] for x in X], axis=1)
        C_n = 2 * (np.log2(self.max_samples - 1) + 0.5772156649) - 2 * (self.max_samples - 1) / self.max_samples
        return 2 ** (-avg_path_lengths / C_n)

    def predict(self, X, threshold=0.5):
        return (self.anomaly_score(X) > threshold).astype(int)

np.random.seed(42)

# For .mat files
#data = loadmat('shuttle.mat')
#X = np.array(data['X'], dtype=np.float64)  # Ensure X is float64
#y = np.array(data['y'].ravel(), dtype=np.int64)  # Convert y to a 1D NumPy array with int64

# For .txt files
data = np.loadtxt('cover.txt', delimiter=",")
X = data[:, :-1]  # Assumes numerical data
y = data[:, -1]


# Convert labels: 1 = anomaly, 0 = normal
y = (y == 1).astype(int)

pgif = PGIForest(num_trees=100, max_samples=256, max_depth=8, k=2)

# Train PGIF on the dataset
pgif.fit(X)

# Compute anomaly scores
scores = pgif.anomaly_score(X)


predictions = pgif.predict(X,0.65)
#unique, counts = np.unique(predictions, return_counts=True)
#print(dict(zip(unique, counts)))

print("Anomaly Detection Completed!")
# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()

# Compute AUC
auc_score = roc_auc_score(y, scores)

# Compute Precision and Recall
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)

# Print results
#print(f"True Positives: {tp}")
#print(f"False Positives: {fp}")
#print(f"True Negatives: {tn}")
#print(f"False Negatives: {fn}")
print(f"AUC: {auc_score:.4f}")
#print(f"Precision: {precision:.4f}")
#print(f"Recall: {recall:.4f}")