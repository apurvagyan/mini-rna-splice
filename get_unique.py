import pandas as pd

# Load the TSV file into a pandas DataFrame
df = pd.read_csv('230531_PSI_quantifications.tsv', sep='\t')

# Extract the part of the string before 'r' character
df['sample_id'] = df['sample_id'].str.split('r').str[0]

# Drop duplicate rows based on 'event_id' and 'sample_id'
df_unique = df.drop_duplicates(subset=['event_id', 'sample_id'])

# Group by 'event_id' and get the counts of unique 'sample_id'
counts = df_unique.groupby('event_id')['sample_id'].nunique().sort_values(ascending=False)

# Print the counts in descending order
print(counts)


# Load a different TSV file into a pandas DataFrame
df2 = pd.read_csv('230531_events_coordinates.tsv', sep='\t')

# Count the number of unique 'event_id'
unique_event_ids = df2['event_id'].nunique()
# Count the number of unique 'gene_id'
unique_gene_ids = df2['gene_id'].nunique()
# Print the counts for df2

df2_counts = df2['event_id'].value_counts()
print(df2_counts)
print("Number of unique event_id:", unique_event_ids)
print("Number of unique gene_id:", unique_gene_ids)