import torch
from scatter import Scatter
from tqdm import tqdm
import pickle 

# def run_scattering():
#     graph_data = torch.load('data/graph_data_matched.pt')
#     coeffs = []
#     for graph in tqdm(graph_data):
#         # print(i)
#         in_channels = graph.x.size(0)
#         max_graph_size = graph.x.size(0)

#         scattering = Scatter(in_channels, max_graph_size)
#         scatter_coeffs = scattering(graph)
#         coeffs.append(scatter_coeffs)
#     # import pdb; pdb.set_trace()
#     # Save the scatter coefficients
#     torch.save(coeffs, 'scatter_coeffs_matched_psi.pt')

def run_scattering_pkl():

    with open('data/gene_ids_matched.txt', 'r') as f:
        gene_ids = [line.strip() for line in f]
    

    # coeffs = []

    with open('data/graph_data_matched.pkl', 'rb') as f:
        for i in tqdm(range(len(gene_ids))):
            graph = pickle.load(f)

            in_channels = graph.x.size(0)
            max_graph_size = graph.x.size(0)

            scattering = Scatter(in_channels, max_graph_size)
            scatter_coeffs = scattering(graph)
            torch.save(scatter_coeffs, f'scatter/scatter_coeffs_matched_psi_{gene_ids[i]}.pt')
            # coeffs.append(scatter_coeffs)

    # torch.save(coeffs, 'scatter_coeffs_matched_psi.pt')

if __name__ == '__main__':
    run_scattering_pkl()
