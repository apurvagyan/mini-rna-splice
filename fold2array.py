import numpy as np
from tqdm import tqdm
import psutil

import sys
sys.path.append('../GSAE/gsae/data_processing')

from utils import dot2adj

# Initialize empty lists to store data
sequences = []
min_free_energies = []

input_filename = 'folded_sequences.txt'
output_filename = 'output'

seq_fname = "{}_sequences.txt".format(output_filename)
adj_mat_fname = "{}_adjmats.npy".format(output_filename)
energy_fname = "{}_energies.csv".format(output_filename)


# Open the file
with open('folded_sequences.txt', 'r') as file:
    # Read the file line by line
    lines = file.readlines()
    
    # Iterate over each line
    for i in tqdm(range(0, len(lines), 4)):  # Assuming each record consists of 4 lines
        # Extract sequence, secondary structure, and minimum free energy
        sequence = lines[i].split(': ')[1].strip()
        secondary_structure = lines[i + 1].split(': ')[1].strip()
        min_free_energy = float(lines[i + 2].split(': ')[1].strip())
        
        # Append extracted data to respective lists
        sequences.append(sequence)
        min_free_energies.append(min_free_energy)

        adj_mat = dot2adj(secondary_structure)

        adj_list = adj_matrix_to_adj_list(adj_mat)


        with open(adj_mat_fname, 'ab') as file:  # Open the file in append binary mode
            np.savetxt(file, adj_list)  # Append new_array to the file

        # memory_usage = psutil.Process().memory_info().rss
        # print("Memory Usage ({}):".format(i/4), memory_usage / (1024 ** 3), "GB")

np.savetxt(seq_fname, sequences, delimiter=',')
np.savetxt(energy_fname, min_free_energies, delimiter=',')

# print(adj_mats[0])
# print(adj_lists[0])

def adj_matrix_to_adj_list(adj_matrix):
    num_nodes = len(adj_matrix)
    adj_list = {}

    for i in range(num_nodes):
        adj_list[i] = []
        for j in range(num_nodes):
            if adj_matrix[i][j] != 0:
                adj_list[i].append(j)

    return adj_list
