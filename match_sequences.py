# # Read the sequences from the files
# with open('../mini_splicenn/240226_partial_sequences.fa', 'r') as file1, open('../data/out_success/output_gpu_sequences.txt', 'r') as file2:
#     sequences1 = [line.strip() for i, line in enumerate(file1) if i % 2 == 1]
#     sequences2 = [line.strip() for i, line in enumerate(file2) if i % 2 == 1][:55]

#     # Print the first line in each sequences object

# # Remove any leading or trailing whitespace from the sequences
# sequences1 = [sequence.strip() for sequence in sequences1]
# sequences2 = [sequence.strip() for sequence in sequences2]

# # Check if the sequences are in the same order
# if sequences1 == sequences2:
#     print("The sequences are in the same order.")
# else:
#     # Find the index where the sequences differ
#     index = next(i for i, (seq1, seq2) in enumerate(zip(sequences1, sequences2)) if seq1 != seq2)
#     print(f"The sequences differ at index {index}.")
#     print("The sequences are not in the same order.")
#     print("The differing sequences are:")
#     print(f"Sequence 1: {sequences1[index]}")
#     print(f"Sequence 2: {sequences2[index]}")
#     index = next(i for i, (seq1, seq2) in enumerate(zip(sequences1, sequences2)) if seq1 != seq2)
#     print(f"The sequences differ at index {index}.")
#     print("The sequences are not in the same order.")


