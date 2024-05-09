from Bio import SeqIO

import matplotlib.pyplot as plt

# Function to parse the file and extract sequence lengths
def parse_file(file_path, max_length=None):
    sequence_lengths = []
    with open(file_path, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            if max_length is None or len(record.seq) <= max_length:
                sequence_lengths.append(len(record.seq))
    return sequence_lengths

# Function to plot the distribution of sequence lengths
def plot_distribution(sequence_lengths, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of RNA Sequence Lengths (Total Samples: {len(sequence_lengths)})')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    if save_path:
        plt.savefig(save_path)  # Save the plot if save_path is provided
    else:
        plt.show()


# Main function
def main():
    file_path = "240507_partial_sequences_match_230531_events.fa"  # Replace with the path to your file
    sequence_lengths = parse_file(file_path)
    plot_distribution(sequence_lengths, 'figs/new_dist.png')

    # Add second distribution with max length 1000
    sequence_lengths_max_1000 = parse_file(file_path, max_length=1000)
    plot_distribution(sequence_lengths_max_1000, 'figs/new_dist_max_1000.png')
    # Add third distribution with max length 10000
    sequence_lengths_max_10000 = parse_file(file_path, max_length=10000)
    plot_distribution(sequence_lengths_max_10000, 'figs/new_dist_max_10000.png')
if __name__ == "__main__":
    main()
