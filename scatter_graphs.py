import torch
from scatter import Scatter
from tqdm import tqdm

def run_scattering():
    graph_data = torch.load('graph_data_under_200.pt')
    coeffs = []
    for graph in tqdm(graph_data):
        # print(i)
        in_channels = graph.x.size(0)
        max_graph_size = graph.x.size(0)

        scattering = Scatter(in_channels, max_graph_size)
        scatter_coeffs = scattering(graph)
        coeffs.append(scatter_coeffs)
    # import pdb; pdb.set_trace()
    # Save the scatter coefficients
    torch.save(coeffs, 'scatter_coeffs_under_200.pt')

if __name__ == '__main__':
    run_scattering()
