import pickle as pk
import numpy as np
from scatter import Scatter
from scipy.sparse import coo_matrix
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import psutil

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


def get_memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 / 1024  # in MB
    return mem


# graph_lst = []
# with open('240507/outputs_240507_2_adjacency.pkl', 'rb') as in_f:
    
#     # graph_lst = []
#     #Replace the 46925 with exact number of sequences in the file
#     for i in (tqdm(range(2465))):
#         data = pk.load(in_f)

#         # if len(data) > 200:
#         #     continue

#         # sequences.append(data)
#         matrix = adj_list_to_adj_matrix(data)
#         # Convert adjacency matrix to edge index
#         edge_index = adjacency_matrix_to_edge_index(matrix)

#         # Assuming no additional node features, initialize node features with one-hot encoding
#         num_nodes = len(data)
#         # print(num_nodes)
#         x = torch.eye(num_nodes)

#         # Create a PyTorch Geometric Data object
#         graph_data = Data(x=x, edge_index=edge_index)
#         # print(graph_data)
#         graph_lst.append(graph_data)
        # torch.save(graph_data, 'graph_data_under_200.pt')

        # Process the graph data to produce scattering coefficients
        # in_channels = graph_data.x.size(0)
        # max_graph_size = graph_data.x.size(0)
        # scattering = Scatter(in_channels, max_graph_size)
        # scatter_coeffs = scattering(graph_data)

        # torch.save(scatter_coeffs, out_f)
    

        # pbar.set_description(f"Memory usage: {get_memory_usage()} MB")


# Save the graph data
# torch.save(graph_lst, 'graph_data.pt')


# graph_data = torch.load('graph_data_under_200.pt')
# print(len(graph_data))

# # coeffs = []
# for graph in tqdm(graph_data):
#     # print(i)
#     in_channels = graph.x.size(0)
#     max_graph_size = graph.x.size(0)
#     scattering = Scatter(in_channels, max_graph_size)
#     scatter_coeffs = scattering(graph)

#     # coeffs.append(scatter_coeffs)

#     with open('scatter_coeffs_under_200.pkl', 'ab') as out_f:
#         torch.save(scatter_coeffs, out_f)

# import pdb; pdb.set_trace()
# Save the scatter coefficients
# torch.save(coeffs, 'scatter_coeffs_200.pt')
