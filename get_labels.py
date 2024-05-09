
from Bio import SeqIO
from tqdm import tqdm
# Path to the FASTA file
fasta_file = "240226_partial_sequences.fa"

# Dictionary to store sequences and gene IDs
sequence_gene_dict = {}

# Read the FASTA file and populate the dictionary
for record in SeqIO.parse(fasta_file, "fasta"):
    # Extracting sequence
    sequence = str(record.seq)
    # Extracting gene ID from the header
    gene_id = record.id
    if len(sequence) > 30000:
        sequence = sequence[:30000]
    # Adding sequence and gene ID to the dictionary
    sequence_gene_dict[sequence] = gene_id

with open('../data/out_success/output_gpu_sequences.txt') as f, open("../data/out_success/gene_ids_under_5000.txt", 'a+') as out_f:
    for line in tqdm(f.readlines()):

        # print(len(line))
        # import pdb; pdb.set_trace()

        if len(line) < 1000:
            out_f.write(f"{sequence_gene_dict[line.strip()]}\n")



