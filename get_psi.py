import pandas as pd

# Read the events coordinates file
events_coordinates = pd.read_csv('230531_events_coordinates.tsv', sep='\t')

# Read the PSI quantifications file
psi_quantifications = pd.read_csv('230531_PSI_quantifications.tsv', sep='\t')

# Merge the two dataframes based on the event_id column
merged_data = pd.merge(events_coordinates, psi_quantifications, on='event_id')

# Create a dictionary of gene_id to psi
gene_id_to_psi = dict(zip(merged_data['gene_id'], merged_data['PSI']))

# Print the gene_id to psi dictionary
print(gene_id_to_psi)