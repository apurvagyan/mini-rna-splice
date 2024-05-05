import os
import RNA
from Bio import SeqIO
from multiprocessing import Pool, cpu_count

OUTPUT_FILE = "results_3.txt"
DATA_PATH = '240226_partial_sequences_copy.fa'

def process_sequence(sequence):
    fc = RNA.fold_compound(sequence)
    (ss, mfe) = fc.mfe()
    return f'Sequence: {sequence}\nSecondary Structure: {ss}\nMinimum Free Energy (MFE): {mfe:.2f}\n\n'

def main():
    with open(OUTPUT_FILE, 'w') as f:
        sequences = [str(record.seq) for record in SeqIO.parse(DATA_PATH, "fasta")]
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(process_sequence, sequences, chunksize=10)
            for result in results:
                f.write(result)

if __name__ == "__main__":
    main()

