import networkx as nx
import matplotlib.pyplot as plt
import random

# Ex 2
# Ex 2 Sub 1 - Combine two graphs
# Generate the regular graph
regular_graph = nx.random_regular_graph(d=3, n=100, seed=42)

# Generate the graph of cliques
clique_graph = nx.connected_caveman_graph(10, 20)  # 10 cliques, each with 20 nodes

# Merge graphs
combined_graph = nx.union(regular_graph, clique_graph, rename=("reg-", "clique-"))

# Add random edges to ensure the graph is connected
random_edges = 50
nodes = list(combined_graph.nodes())
for _ in range(random_edges):
    u = random.choice(nodes)
    v = random.choice(nodes)
    if u != v and not combined_graph.has_edge(u, v):
        combined_graph.add_edge(u, v)

print(f"Combined graph has {combined_graph.number_of_nodes()} nodes and {combined_graph.number_of_edges()} edges.")

for node in combined_graph.nodes():
    ego = nx.ego_graph(combined_graph, node)

    Ni = ego.number_of_nodes() - 1  # Exclude the node itself
    Ei = ego.size()

    combined_graph.nodes[node]["Ni"] = Ni
    combined_graph.nodes[node]["Ei"] = Ei

# Compute anomaly scores based on Ei and Ni
clique_scores = {}
for node in combined_graph.nodes(data=True):
    Ni = node[1]["Ni"]
    Ei = node[1]["Ei"]
    clique_scores[node[0]] = Ni + Ei

# Sort nodes by clique scores in descending order
sorted_clique_nodes = sorted(clique_scores, key=clique_scores.get, reverse=True)

# Highlight the top 10 nodes most likely part of cliques
top_clique_nodes = sorted_clique_nodes[:10]

# Assign colors to nodes
node_colors = []
for node in combined_graph.nodes():
    if node in top_clique_nodes:
        node_colors.append("red")  # Top nodes are red
    else:
        node_colors.append("blue")  # The rest are blue

plt.figure(figsize=(12, 12))
pos = nx.spring_layout(combined_graph, seed=42)
nx.draw(
    combined_graph,
    pos,
    with_labels=False,
    node_color=node_colors,
    node_size=50,
    edge_color="gray",
    alpha=0.7,
)
plt.title("Graph with Top 10 Nodes Most Likely Part of Cliques Highlighted", fontsize=16)
plt.show()

# Ex 2 Sub 2
import networkx as nx
import matplotlib.pyplot as plt
import random

graph1 = nx.random_regular_graph(d=3, n=100, seed=42)
graph2 = nx.random_regular_graph(d=5, n=100, seed=43)

merged_graph = nx.union(graph1, graph2, rename=("g1-", "g2-"))

for u, v in merged_graph.edges():
    merged_graph[u][v]["weight"] = 1

random_nodes = random.sample(list(merged_graph.nodes()), 2)
for node in random_nodes:
    egonet = nx.ego_graph(merged_graph, node)
    for u, v in egonet.edges():
        merged_graph[u][v]["weight"] += 10

anomaly_scores = {}
for node in merged_graph.nodes():
    egonet = nx.ego_graph(merged_graph, node)
    E_i = egonet.size()
    W_i = sum(data["weight"] for _, _, data in egonet.edges(data=True))
    anomaly_scores[node] = W_i + E_i

# Get the top 4 nodes with the greatest scores
top_anomalous_nodes = sorted(anomaly_scores, key=anomaly_scores.get, reverse=True)[:4]

# Assign colors to nodes
node_colors = []
for node in merged_graph.nodes():
    if node in top_anomalous_nodes:
        node_colors.append("red")  # Anomalies are red
    else:
        node_colors.append("blue")  # Normal data points are blue

# Draw the merged graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(merged_graph, seed=42)
nx.draw(
    merged_graph,
    pos,
    with_labels=False,
    node_color=node_colors,
    node_size=50,
    edge_color="gray",
    alpha=0.7,
)
plt.title("Graph with HeavyVicinity Anomalies Highlighted", fontsize=16)
plt.show()