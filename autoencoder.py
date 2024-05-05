import torch

import torch.nn as nn

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
        return encoded,decoded


if __name__ == '__main__':
    input_dim = 10
    latent_dim = 3
    model = Autoencoder(input_dim, latent_dim)
    print(model)
    # Load the scatter coefficients
    scatter_coeffs = torch.load('scatter_coeffs.pt')
    import pdb; pdb.set_trace()
    # Pass the scatter coefficients through the model
    encoded, decoded = model(scatter_coeffs)

    # Save the latent space representation
    torch.save(encoded, 'latent_space.pt')