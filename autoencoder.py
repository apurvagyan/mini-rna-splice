# import torch
# import os
# import torch.nn as nn
# import torch.optim as optim
# from torch_geometric.nn import global_mean_pool
# from tqdm import tqdm
# import glob
# import pandas as pd
# import pickle as pk
# import multiprocessing
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super(Autoencoder, self).__init__()
        
#         # Encoder layers
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, latent_dim)
#         )
        
#         # Decoder layers
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, input_dim),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):        
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded

# def pad_coefficient_matrices(coefficients, max_num_nodes):
#     padded_coefficients = []
#     for coefficient_matrix in tqdm(coefficients):
#         num_nodes = min(max_num_nodes, coefficient_matrix.shape[0])
#         padded_coefficient_matrix = torch.zeros(max_num_nodes, 11, max_num_nodes)
#         padded_coefficient_matrix[:num_nodes, :, :num_nodes] = coefficient_matrix[:num_nodes, :, :num_nodes]
#         padded_coefficients.append(padded_coefficient_matrix)
#     padded_coefficients = torch.stack(padded_coefficients, dim=0)
#     return padded_coefficients

# if __name__ == '__main__':
#     # print(torch.cuda.is_available())
#     # exit()
#     device = 'cuda'
#     model_name = 'model_new.pt'
#     scatter_all = []
#     # scatter_coeffs = torch.load('scatter_coeffs_new.pt')
#     with open('scatter_coeffs_new.pkl', 'rb') as f:
#         for _ in tqdm(range(2465)):
#             scatter_coeffs = pk.load(f)
#             scatter_all.append(scatter_coeffs)
    
    
#     # #slice for the geneid part onl
#     # all_scatter_coeffs = []
#     # all_gene_ids = []
#     # import pdb; pdb.set_trace()
#     # # import pdb; pdb.set_trace()
#     # for i in tqdm(scatter_file_list[:100]):
#     #     #Slice for getting the geneID
#     #     gene_id = i.split('_')[-1].split('.')[0]
#     #     all_gene_ids.append(gene_id)
#     #     # import pdb; pdb.set_trace()
#     #     scatter_coeffs = torch.load(i)
#     #     all_scatter_coeffs.append(scatter_coeffs)
    
#     # # Load the labels.csv file
#     # labels_df = pd.read_csv('scatter/labels.csv')

#     # # Create a dictionary to map gene IDs to psi values
#     # geneid_to_psi = dict(zip(labels_df['gene_id'], labels_df['PSI']))

#     # # Get the psi values for each gene ID
#     # psi_values = [geneid_to_psi[gene_id] for gene_id in all_gene_ids]

    

#     max_num_nodes = max([item.shape[0] for item in scatter_coeffs])

#     padded_coefficients = pad_coefficient_matrices(scatter_coeffs, max_num_nodes)
    
#     input_dim = max_num_nodes
#     latent_dim = 64

#     if os.path.exists(model_name):
#         model = Autoencoder(input_dim, latent_dim)
#         model.load_state_dict(torch.load(model_name))
#     else:

#         model = Autoencoder(input_dim, latent_dim).to(device)
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001)


#         # Run the training code here
#         train_size = int(0.8 * len(padded_coefficients))
#         test_size = len(padded_coefficients) - train_size

#         train_data, test_data = torch.utils.data.random_split(padded_coefficients, [train_size, test_size])
        
#         batch_size = 64
#         train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
#         # Evaluate the model on the test set
#         test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
#         num_epochs = 10

#         # Convert psi values to tensor
#         # psi_values_tensor = torch.tensor(psi_values, dtype=torch.float32)

#         # # Normalize psi values between 0 and 1
#         # psi_values_tensor = (psi_values_tensor - psi_values_tensor.min()) / (psi_values_tensor.max() - psi_values_tensor.min())

#         # # Split the data into train and test sets
#         # train_size = int(0.8 * len(padded_coefficients))
#         # test_size = len(padded_coefficients) - train_size
#         # train_psi_values, test_psi_values = psi_values_tensor[:train_size], psi_values_tensor[train_size:]

#         # Define the loss function and optimizer
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001)

#         # Train the model to predict psi values from scattering coefficients
#         # num_epochs = 10
#         # for epoch in range(num_epochs):
#         #     train_loss = 0.0
#         #     for batch, psi_values_batch in tqdm(zip(train_loader, train_psi_values)):
#         #         batch = batch.to(device)
#         #         psi_values_batch = psi_values_batch.to(device)
#         #         optimizer.zero_grad()
#         #         encoded, decoded = model(batch)
#         #         loss = criterion(decoded, batch) + criterion(encoded, psi_values_batch.unsqueeze(1))
#         #         loss.backward(retain_graph=True)
#         #         optimizer.step()
#         #         train_loss += loss.item()
#         #     train_loss /= len(train_data)
#         #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')

        
#         # test_loss = 0.0
#         # with torch.no_grad():
#         #     for batch, psi_values_batch in tqdm(zip(test_loader, test_psi_values)):
#         #         batch = batch.to(device)
#         #         psi_values_batch = psi_values_batch.to(device)
#         #         encoded, decoded = model(batch)
#         #         loss = criterion(decoded, batch) + criterion(encoded, psi_values_batch.unsqueeze(1))
#         #         test_loss += loss.item()
#         # test_loss /= len(test_data)
#         # print(f'Test Loss: {test_loss:.4f}')

#         for epoch in range(num_epochs):
#             train_loss = 0.0
#             for batch in tqdm(train_loader):
#                 # import pdb; pdb.set_trace()
#                 batch = batch.to(device)
#                 optimizer.zero_grad()
#                 encoded, decoded = model(batch)
#                 loss = criterion(decoded, batch)
#                 loss.backward(retain_graph=True)
#                 optimizer.step()
#                 train_loss += loss.item()
#             train_loss /= len(train_data)
#             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')

#         # Save the latent space representation
#         torch.save(model.state_dict(), model_name)


#     # Get the latent space representation
#     latent_space = []  
#     for sample in padded_coefficients:
#         encoded, _ = model(sample)
#         latent_space.append(encoded)

#     latent_space = torch.stack(latent_space, dim=0)

#     # Save the latent space representation

#     torch.save(latent_space, 'latent_space_new.pt')

import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
import glob
import pandas as pd
import pickle as pk
import multiprocessing
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

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

def pad_coefficient_matrix(coefficient_matrix, max_num_nodes):
    num_nodes = min(max_num_nodes, coefficient_matrix.shape[0])
    padded_coefficient_matrix = torch.zeros(max_num_nodes, 11, max_num_nodes)
    padded_coefficient_matrix[:num_nodes, :, :num_nodes] = coefficient_matrix[:num_nodes, :, :num_nodes]
    return padded_coefficient_matrix

def run_padding_in_parallel(coefficients, max_num_nodes):
    with Pool(processes=cpu_count()) as pool:
        padded_coefficients = list(tqdm(pool.imap(partial(pad_coefficient_matrix, max_num_nodes=max_num_nodes), coefficients), total=len(coefficients)))
    padded_coefficients = torch.stack(padded_coefficients, dim=0)
    return padded_coefficients

def load_scatter_chunk(filename, start, end):
    scatter_all = []
    with open(filename, 'rb') as f:
        for _ in range(start):
            pk.load(f)  # Skip the first 'start' elements
        for _ in range(start, end):
            try:
                scatter_coeffs = pk.load(f)
                scatter_all.append(scatter_coeffs)
            except EOFError:
                break
    return scatter_all

def run_loading_in_parallel(filename, total_items, chunk_size):
    pool = Pool(processes=cpu_count())
    tasks = []
    for i in range(0, total_items, chunk_size):
        start = i
        end = min(i + chunk_size, total_items)
        tasks.append((filename, start, end))

    scatter_all = []
    for result in tqdm(pool.starmap(load_scatter_chunk, tasks), total=len(tasks)):
        scatter_all.extend(result)

    pool.close()
    pool.join()
    return scatter_all

def main():
    device = 'cuda'
    model_name = 'model_new.pt'
    scatter_all = []
    
    # Parallel load scatter coefficients
    filename = 'scatter_coeffs_new.pkl'
    total_items = 2465
    chunk_size = 100  # Adjust based on your memory and performance needs
    scatter_all = run_loading_in_parallel(filename, total_items, chunk_size)
    
    max_num_nodes = max([item.shape[0] for item in scatter_all])
    
    # Run padding in parallel
    padded_coefficients = run_padding_in_parallel(scatter_all, max_num_nodes)
    
    input_dim = max_num_nodes
    latent_dim = 64

    if os.path.exists(model_name):
        model = Autoencoder(input_dim, latent_dim)
        model.load_state_dict(torch.load(model_name))
    else:
        model = Autoencoder(input_dim, latent_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Run the training code here
        train_size = int(0.8 * len(padded_coefficients))
        test_size = len(padded_coefficients) - train_size

        train_data, test_data = torch.utils.data.random_split(padded_coefficients, [train_size, test_size])
        
        batch_size = 64
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        num_epochs = 10

        for epoch in range(num_epochs):
            train_loss = 0.0
            for batch in tqdm(train_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                encoded, decoded = model(batch)
                loss = criterion(decoded, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_data)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')

        # Save the model state
        torch.save(model.state_dict(), model_name)

    # Get the latent space representation
    latent_space = []  
    for sample in padded_coefficients:
        sample = sample.to(device)
        encoded, _ = model(sample)
        latent_space.append(encoded)

    latent_space = torch.stack(latent_space, dim=0)

    # Save the latent space representation
    torch.save(latent_space, 'latent_space_new.pt')

if __name__ == '__main__':
    main()



