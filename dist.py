import matplotlib.pyplot as plt

# Path to the input file
input_file = '../data/out_success/output_gpu_sequences.txt'

# Read the lines from the file
with open(input_file, 'r') as file:
    lines = file.readlines()

# Get the length of each line
line_lengths = [len(line.strip()) for line in lines]

# Create the histogram
plt.hist(line_lengths, bins=10, edgecolor='black')

# Set the labels and title
plt.xlabel('Sequence Length')
plt.ylabel('Count')
plt.title('Distribution of Sequence Lengths')

# Display the histogram
plt.show()
# Save the figure as PNG
plt.savefig('line_lengths.png')

# Filter line lengths that are less than or equal to 1000
filtered_lengths = [length for length in line_lengths if length <= 1000]


# Create a new figure for the second histogram
plt.figure()

# Create the histogram with filtered lengths
plt.hist(filtered_lengths, bins=10, edgecolor='black')


# Set the labels and title
plt.xlabel('Sequence Length')
plt.ylabel('Count')
plt.title('Distribution of Sequence Lengths (Up to 1000 BP)')


# Display the histogram
plt.show()


# Save the figure as PNG
plt.savefig('line_lengths_up_to_1000.png')