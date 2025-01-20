import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

# Ex 1
# Ex 1 Sub 1 - Load dataset, build graph
file_path = "ca-AstroPh.txt"
G = nx.Graph()

with open(file_path, "r") as f:
    for i, line in enumerate(f):
        if i < 4 or i == 197065:    # First 4 lines are useless, as is the last one
            continue
        node1, node2 = map(int, line.strip().split())
        if G.has_edge(node1, node2):
            # Increment edge weight if the edge already exists
            G[node1][node2]["weight"] += 1
        else:
            G.add_edge(node1, node2, weight=1)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges (only counting distinct ones)")

# Ex 1 Sub 2 - Extract features from egonet
features = {}

for node in G.nodes():
    ego = nx.ego_graph(G, node)

    Ni = ego.number_of_nodes() - 1  # Exclude the node itself
    Ei = ego.size()
    Wi = sum(data['weight'] for _, _, data in ego.edges(data=True))
    adjacency_matrix = nx.to_numpy_array(ego, weight="weight")
    eigenvalues = np.linalg.eigh(adjacency_matrix)[0]
    lambda_w = max(eigenvalues)

    features[node] = {
        "Ni": Ni,
        "Ei": Ei,
        "Wi": Wi,
        "lambda_w": lambda_w,
    }

nx.set_node_attributes(G, features)
#sample_node = list(G.nodes())[0]   # To check that it works
#print(f"Features for node {sample_node}: {G.nodes[sample_node]}") # To check that it works

# Ex 1 Sub 3 - Compute anomaly score for nodes with linear regression
log_Ni = []
log_Ei = []

# Extract log(Ni) and log(Ei) for each node
for node, attributes in G.nodes(data=True):
    log_Ni.append(np.log(attributes["Ni"] + 1))
    log_Ei.append(np.log(attributes["Ei"] + 1))

log_Ni = np.array(log_Ni).reshape(-1, 1)
log_Ei = np.array(log_Ei)

# Fit Linear Regression model
reg = LinearRegression()
reg.fit(log_Ni, log_Ei)

C = reg.intercept_
theta = reg.coef_[0]

# Compute anomaly scores
scores = {}
for node, attributes in G.nodes(data=True):
    yi = attributes["Ei"]
    xi = attributes["Ni"]
    predicted_y = C * (xi ** theta)  # C * x^θ

    # Calculate the anomaly score
    score = (max(yi, predicted_y) / min(yi, predicted_y)) * np.log(abs(yi - predicted_y) + 1)

    scores[node] = score

# Ex 1 Sub 4 - Sort nodes in descending order and draw graph
sorted_nodes = sorted(scores, key=scores.get, reverse=True)

# Store the scores in the graph
nx.set_node_attributes(G, scores, "anomaly_score")

#print(scores)

subgraph_nodes = list(G.nodes())[:1500]
subgraph = G.subgraph(subgraph_nodes)

# Get the top 10 anomalous nodes in the subgraph
top_10_nodes = [node for node in sorted_nodes if node in subgraph][:10]

# Assign colors to the nodes
node_colors = []
for node in subgraph:
    if node in top_10_nodes:
        node_colors.append("red")  # Top 10 nodes are red
    else:
        node_colors.append("blue")  # Others are blue

# Draw the graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(subgraph, seed=42)
nx.draw(
    subgraph,
    pos,
    with_labels=False,
    node_color=node_colors,
    node_size=50,
    edge_color="gray",
    alpha=0.7,
)
plt.title("Graph with Top 10 Anomalous Nodes Highlighted", fontsize=16)
plt.show()

# Ex 1 Sub 5 - Modify anomaly score and redraw graph
Ei_Ni_features = np.array([[attributes["Ei"], attributes["Ni"]] for _, attributes in G.nodes(data=True)])

lof = LocalOutlierFactor(n_neighbors=20, metric="euclidean")
lof_scores = -lof.fit_predict(Ei_Ni_features)

# Normalize LOF scores
scaler = MinMaxScaler()
lof_scores_normalized = scaler.fit_transform(lof_scores.reshape(-1, 1)).flatten()

new_scores = {}
for i, node in enumerate(G.nodes()):
    original_score = scores[node]
    lof_score = lof_scores_normalized[i]

    new_scores[node] = original_score + lof_score

sorted_new_nodes = sorted(new_scores, key=new_scores.get, reverse=True)

nx.set_node_attributes(G, new_scores, "modified_anomaly_score")

top_10_new_nodes = [node for node in sorted_new_nodes if node in subgraph][:10]

# Assign colors to the nodes
new_node_colors = []
for node in subgraph:
    if node in top_10_new_nodes:
        new_node_colors.append("green")  # Top 10 highest wtih anomaly score are green
    else:
        new_node_colors.append("blue")  # The rest are blue

# Draw the graph with modified anomaly scores
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(subgraph, seed=42)
nx.draw(
    subgraph,
    pos,
    with_labels=False,
    node_color=new_node_colors,
    node_size=50,
    edge_color="gray",
    alpha=0.7,
)
plt.title("Graph with Top 10 Anomalous Nodes (Modified Scores) Highlighted", fontsize=16)
plt.show()
