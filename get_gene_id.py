import csv
import glob
import pandas as pd

# Get a list of all the .fa files in the directory
file_list = glob.glob('../mini_splicenn/mini-rna-splice/240507_partial_sequences_match_230531_events.fa')

# Set to store unique strings
unique_strings = set()

# Iterate over each file
for file_path in file_list:
    with open(file_path, 'r') as file:
        # Read each line in the file
        for line in file:
            # Check if the line starts with '>'
            if line.startswith('>'):
                # print(line)
                # Extract the string after '>'
                string = line[1:].strip()
                # Add the string to the set
                unique_strings.add(string)

# Print the number of unique strings
print(len(unique_strings))


# Read the TSV file into a pandas DataFrame
df = pd.read_csv('230531_events_coordinates.tsv', sep='\t')

# Get the unique entries in the 'gene_id' column
unique_gene_ids = df['gene_id'].unique()

# Print the number of unique gene IDs
print(len(unique_gene_ids))