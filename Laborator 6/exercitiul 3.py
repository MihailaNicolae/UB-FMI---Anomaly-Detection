import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from scipy.io import loadmat
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score
from io import BytesIO

# Ex 3
# Ex 3 Sub 1 + 3 - Graph Autoencoder (GAE)
class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(GraphAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim)
        self.attr_decoder = AttributeDecoder(input_dim)
        self.struct_decoder = StructureDecoder()

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)

        x_hat = self.attr_decoder(z, edge_index)

        a_hat = self.struct_decoder(z, edge_index)

        return x_hat, a_hat, z

class Encoder(nn.Module):
    def __init__(self, input_dim):
        super(Encoder, self).__init__()
        self.gcn1 = pyg_nn.GCNConv(input_dim, 128)
        self.gcn2 = pyg_nn.GCNConv(128, 64)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index).relu()
        z = self.gcn2(x, edge_index).relu()
        return z

class AttributeDecoder(nn.Module):
    def __init__(self, output_dim):
        super(AttributeDecoder, self).__init__()
        self.gcn1 = pyg_nn.GCNConv(64, 128)
        self.gcn2 = pyg_nn.GCNConv(128, output_dim)

    def forward(self, z, edge_index):
        z = self.gcn1(z, edge_index).relu()
        x_hat = self.gcn2(z, edge_index)
        return x_hat

class StructureDecoder(nn.Module):
    def __init__(self):
        super(StructureDecoder, self).__init__()
        self.gcn = pyg_nn.GCNConv(64, 64)

    def forward(self, z, edge_index):
        z = self.gcn(z, edge_index).relu()
        a_hat = torch.matmul(z, z.T)
        return a_hat

# Ex 3 Sub 2 - load ACM dataset
#data = loadmat('ACM.mat') # For some reason, this did not work

file_path = "ACM2.mat"
with open(file_path, 'rb') as f:
    content = f.read()

mat_file = BytesIO(content)
data = loadmat(mat_file)

attributes = torch.tensor(data["Attributes"].toarray(), dtype=torch.float32)
adjacency_matrix = data["Network"]
labels = torch.tensor(data["Label"].flatten(), dtype=torch.long)

edge_index, edge_weight = from_scipy_sparse_matrix(adjacency_matrix)

# Ex 3 Sub 4 - Loss function
def custom_loss(X, X_hat, A, A_hat, alpha=0.8):
    attr_loss = torch.norm(X - X_hat, p="fro") ** 2

    struct_loss = torch.norm(A - A_hat, p="fro") ** 2

    loss = alpha * attr_loss + (1 - alpha) * struct_loss
    return loss

# Ex 3 Sub 5 - Training procedure

# Initialize the model, optimizer, and hyperparameters
input_dim = attributes.shape[1]
model = GraphAutoencoder(input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
alpha = 0.8
num_epochs = 50

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()  # Reset gradients

    X_hat, A_hat, Z = model(attributes, edge_index)

    adjacency_matrix_dense = torch.tensor(adjacency_matrix.todense(), dtype=torch.float32)

    loss = custom_loss(attributes, X_hat, adjacency_matrix_dense, A_hat, alpha)

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            reconstruction_error = torch.norm(attributes - X_hat, p=2, dim=1).numpy()
            auc_score = roc_auc_score(labels.numpy(), reconstruction_error)
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, ROC AUC = {auc_score:.4f}")

print("Training complete.")