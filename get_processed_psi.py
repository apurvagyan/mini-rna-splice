import pickle as pk
from tqdm import tqdm
import pandas as pd
import os


processed_genes = []
processed_genes_file = 'processed_genes.txt'
if not os.path.exists(processed_genes_file):
    with open('../data/out_success/output_gpu_adjacency.pkl', 'rb') as in_f, open('../data/out_success/gene_ids.txt', 'r') as in_gene_ids:
        for i in (pbar := tqdm(range(46925))):
            data = pk.load(in_f)
            gene_id = in_gene_ids.readline().strip()

            if len(data) < 200:
                processed_genes.append(gene_id)

    with open(processed_genes_file, 'w') as out_f:
        for gene_id in processed_genes:
            out_f.write(gene_id + '\n')

else:
    with open(processed_genes_file, 'r') as in_f:
        processed_genes = [line.strip() for line in in_f]



tsv_file = '230531_events_coordinates.tsv'
# Load the TSV file as a pandas DataFrame
df = pd.read_csv(tsv_file, sep='\t')

sorted_df = df.sort_values('gene_length')
print(sorted_df)


matching_rows = df[df['gene_id'].isin(processed_genes)]
exit()

if not matching_rows.empty:
    print("There are matching genes:")
    print(matching_rows)

    matching_rows_unique_gene_id = matching_rows.groupby('gene_id').filter(lambda x: len(x) == 1)

    event_ids = matching_rows_unique_gene_id['event_id'].tolist()

    psi_file = '230531_PSI_quantifications.tsv'
    psi_df = pd.read_csv(psi_file, sep='\t')
    filtered_df = psi_df[psi_df['event_id'].isin(event_ids)]

    filtered_df.drop('nb_reads', axis=1, inplace=True)

    # Calculate average PSI over rows with identical sample_id
    filtered_df['sample_id'] = filtered_df['sample_id'].str.split('r').str[0]
    average_psi = filtered_df.groupby(['event_id', 'sample_id'])['PSI'].mean()

    average_psi_df = average_psi.reset_index()
    # filtered_df = pd.merge(filtered_df, average_psi_df, on=['event_id', 'sample_id'])

    print(average_psi_df)

    output_file = 'processed_psis.csv'
    average_psi_df.to_csv(output_file, index=False)
    print("Filtered DataFrame saved to", output_file)


