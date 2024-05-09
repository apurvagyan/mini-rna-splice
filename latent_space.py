from tqdm import tqdm
import torch
import torch.nn as nn
import os
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded



def pad_coefficient_matrices(coefficients, max_num_nodes):
    
    padded_coefficients = []
    for coefficient_matrix in tqdm(coefficients):
        num_nodes = min(max_num_nodes, coefficient_matrix.shape[0])
        padded_coefficient_matrix = torch.zeros(max_num_nodes, 11, max_num_nodes)
        padded_coefficient_matrix[:num_nodes, :, :num_nodes] = coefficient_matrix[:num_nodes, :, :num_nodes]
        padded_coefficients.append(padded_coefficient_matrix)
    padded_coefficients = torch.stack(padded_coefficients, dim=0)
    return padded_coefficients


device = 'cpu'
model_name = 'model_under_200.pt'

scatter_coeffs = torch.load('scatter_coeffs_under_200.pt')

max_num_nodes = max([item.shape[0] for item in scatter_coeffs])

padded_coefficients = pad_coefficient_matrices(scatter_coeffs[:5000], max_num_nodes)

input_dim = max_num_nodes
latent_dim = 64

if os.path.exists(model_name):
    model = Autoencoder(input_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(model_name))
# Get the latent space representation
latent_space = []  
for sample in padded_coefficients:
    encoded, _ = model(sample.to(device))
    latent_space.append(encoded)

latent_space = torch.stack(latent_space, dim=0)
#Save the latent space representation
torch.save(latent_space, 'latent_space.pt')