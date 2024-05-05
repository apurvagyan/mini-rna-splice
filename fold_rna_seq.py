import os
import RNA
from Bio import SeqIO
from multiprocessing import Pool, cpu_count

OUTPUT_FILE = "folded_sequences.txt"
DATA_PATH = '240226_partial_sequences_copy.fa'

# wrapper function for running Vienna RNAFold
def process_sequence(sequence):
    fc = RNA.fold_compound(sequence)
    (ss, mfe) = fc.mfe()
    return f'Sequence: {sequence}\nSecondary Structure: {ss}\nMinimum Free Energy (MFE): {mfe:.2f}\n\n'

def main():
    # Parameter for maximum sequence length truncation
    max_seq_length = 30000

    with open(OUTPUT_FILE, 'w') as f:
        # read in data
        sequences = [str(record.seq) for record in SeqIO.parse(DATA_PATH, "fasta")]

        longest_seq = max(sequences, key = len)
        shortest_seq = min(sequences, key = len)
        print(f'longest sequence len is {len(longest_seq)}')
        print(f'shortest sequence len is {len(shortest_seq)}')

        print(f'cutting sequences to length {max_seq_length}')
        sequences = [seq[:max_seq_length] for seq in sequences]

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(process_sequence, sequences, chunksize=10)
            for result in results:
                f.write(result)

if __name__ == "__main__":
    main()

