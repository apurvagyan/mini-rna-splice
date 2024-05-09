import torch
import pickle as pk

# # Load the file
# with open('../data/out_success/output_gpu_adjacency.pkl', 'rb') as f:
#     while True:
#         obj = pk.load(f)
#         print(len(obj))

with open('data/graph_data_matched.pkl', 'rb') as f:
    count = 0
    while True:
        try:
            graph = pk.load(f)
            count += 1
        except EOFError:
            break
        # data = pk.load(f)
        # print(data['x'].shape)
print(count)
# coeffs = torch.load('padded_coefficients_matched_psi.pt')

# for coeff in coeffs:
#     print(coeff.shape)