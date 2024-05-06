import pickle as pk
import numpy as np
from scatter import Scatter
from scipy.sparse import coo_matrix
import torch
from torch_geometric.data import Data
from tqdm import tqdm

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
def adjacency_matrix_to_edge_index(adjacency_matrix):
    """
    Convert adjacency matrix to edge index (COO format).
    """
    coo = coo_matrix(adjacency_matrix)
    edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)
    return edge_index

# sequences = []
# with open('out_success/output_gpu_adjacency.pkl', 'rb') as f:
#     graph_lst = []
#     for i in tqdm(range(200)):
#         data = pk.load(f)
#         # sequences.append(data)
#         matrix = adj_list_to_adj_matrix(data)
#         # Convert adjacency matrix to edge index
#         edge_index = adjacency_matrix_to_edge_index(matrix)

#         # Assuming no additional node features, initialize node features with one-hot encoding
#         num_nodes = len(data)
#         x = torch.eye(num_nodes)

#         # Create a PyTorch Geometric Data object
#         graph_data = Data(x=x, edge_index=edge_index)
#         # print(graph_data)
#         graph_lst.append(graph_data)

# Save the graph data
# torch.save(graph_lst, 'graph_data.pt')
graph_data = torch.load('graph_data.pt')
coeffs = []
for graph in tqdm(graph_data[:20]):
    # print(i)
    in_channels = graph.x.size(0)
    max_graph_size = graph.x.size(0)
    scattering = Scatter(in_channels, max_graph_size)
    scatter_coeffs = scattering(graph)
    coeffs.append(scatter_coeffs)
# import pdb; pdb.set_trace()
# Save the scatter coefficients
torch.save(coeffs, 'scatter_coeffs_200.pt')
