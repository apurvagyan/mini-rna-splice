import torch

import phate
import matplotlib.pyplot as plt


latent_reps = torch.load('latent_space.pt')

# print(latent_reps.shape)

average_latent_reps = torch.mean(latent_reps, dim=(1,2)).detach().numpy()
# print(average_latent_reps.shape)
# Create PHATE object
phate_operator = phate.PHATE()

# Fit and transform the latent representation
z_phate = phate_operator.fit_transform(average_latent_reps)

# Visualize the latent representation
plt.figure(figsize=(8, 6))
plt.scatter(z_phate[:, 0], z_phate[:, 1], s=5, cmap='viridis')
plt.xlabel('PHATE 1')
plt.ylabel('PHATE 2')
plt.title('Latent Representations')
plt.show()

plt.savefig('latent_representations.png')
