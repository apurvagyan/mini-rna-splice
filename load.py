import pickle as pk
import numpy as np
from scatter import Scatter
from scipy.sparse import coo_matrix
import torch
from torch_geometric.data import Data

with open('out_success/output_gpu_adjacency.pkl', 'rb') as f:
    data = pk.load(f)
    data_2 = pk.load(f)

print(data)

def adj_list_to_adj_matrix(adj_list):
    """
    Convert an adjacency list to an adjacency matrix.
    
    Parameters:
        adj_list (dict): The adjacency list representing the graph.
    
    Returns:
        numpy.ndarray: The adjacency matrix.
    """
    num_nodes = len(adj_list)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            adj_matrix[node][neighbor] = 1  # or any other value to represent the edge weight

    return adj_matrix

matrix = adj_list_to_adj_matrix(data)
def adjacency_matrix_to_edge_index(adjacency_matrix):
    """
    Convert adjacency matrix to edge index (COO format).
    """
    coo = coo_matrix(adjacency_matrix)
    edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)
    return edge_index


# Convert adjacency matrix to edge index
edge_index = adjacency_matrix_to_edge_index(data)

# Assuming no additional node features, initialize node features with one-hot encoding
num_nodes = len(data)
x = torch.eye(num_nodes)

# Create a PyTorch Geometric Data object
graph_data = Data(x=x, edge_index=edge_index)
print(graph_data)

in_channels = len(data)
max_graph_size = len(data)
scattering = Scatter(in_channels, max_graph_size)
scatter_coeffs = scattering(graph_data)
print(scatter_coeffs)
# Save the scatter coefficients
torch.save(scatter_coeffs, 'scatter_coeffs.pt')
