import os
from Bio import SeqIO
import pandas as pd


gene_ids_file = '../data/out_success/gene_ids_under_5000.txt'
with open(gene_ids_file, 'r') as file:
    gene_ids = set(line.strip() for line in file)

# Read the TSV file and extract gene IDs
tsv_file = '230531_events_coordinates.tsv'
# Load the TSV file as a pandas DataFrame
df = pd.read_csv(tsv_file, sep='\t')

# Filter rows with gene IDs that are in the gene IDs file and start with 'SE'
matching_rows = df[df['gene_id'].isin(gene_ids) & df['event_id'].str.startswith('SE')]
if not matching_rows.empty:
    # print("There are matching genes:")
    # print(matching_rows)

    unique_gene_ids = matching_rows['gene_id'].unique()
    # print(len(unique_gene_ids))


    matching_rows_unique_gene_id = matching_rows.groupby('gene_id').filter(lambda x: len(x) == 1)
    # print(matching_rows_unique_gene_id)

    unique_gene_ids = matching_rows_unique_gene_id['gene_id'].unique()
    # print(len(unique_gene_ids))

    # output_file = 'unique_gene_ids.txt'
    # with open(output_file, 'w') as file:
    #     for gene_id in unique_gene_ids:
    #         file.write(gene_id + '\n')
    # print("Unique gene IDs written to", output_file)

    event_ids = matching_rows_unique_gene_id['event_id'].tolist()

    psi_file = '230531_PSI_quantifications.tsv'
    psi_df = pd.read_csv(psi_file, sep='\t')
    filtered_df = psi_df[psi_df['event_id'].isin(event_ids)]
    # print(filtered_df)

    filtered_df.drop('nb_reads', axis=1, inplace=True)

    # Calculate average PSI over rows with identical sample_id
    filtered_df['sample_id'] = filtered_df['sample_id'].str.split('r').str[0]
    average_psi = filtered_df.groupby(['event_id', 'sample_id'])['PSI'].mean()

    average_psi_df = average_psi.reset_index()
    average_psi_df = pd.merge(average_psi_df, matching_rows_unique_gene_id[['event_id', 'gene_id']], on='event_id', how='left')
    # filtered_df = pd.merge(filtered_df, average_psi_df, on=['event_id', 'sample_id'])

    # print(average_psi_df)
    average_psi_df = average_psi_df[['gene_id', 'event_id', 'sample_id', 'PSI']]
    print(average_psi_df)
    output_file = 'scatter/labels.csv'
    average_psi_df.to_csv(output_file, index=False)
    print("average_psi_df saved to", output_file)
    # Print a row in matching_rows_unique_gene_id that has a certain gene_id
    # gene_id_to_find = 'WBGene00012471'
    # row_to_print = matching_rows_unique_gene_id[matching_rows_unique_gene_id['gene_id'] == gene_id_to_find]
    # print(row_to_print)

    # filtered_df['average_psi'] = average_psi

    # # print(filtered_df)
    sample_id_counts = average_psi_df['sample_id'].value_counts()
    # print(sample_id_counts)


    # output_file = 'test_data_df_under_1000.csv'
    # average_psi_df.to_csv(output_file, index=False)
    # print("Filtered DataFrame saved to", output_file)


else:
    print("No matching genes found.")

