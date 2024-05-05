import pickle

def read_dicts_from_pickle_file(file_path):
    dicts = []
    try:
        with open(file_path, 'rb') as file:
            while True:
                try:
                    # Load a dictionary from the pickle file
                    data = pickle.load(file)
                    dicts.append(data)
                except EOFError:
                    # End of file reached
                    break
    except FileNotFoundError:
        # Handle the case where the file doesn't exist
        print("File not found:", file_path)
    return dicts

# File path
file_path = 'output_adjacency.pkl'

# Read dictionaries from the pickle file
result_dicts = read_dicts_from_pickle_file(file_path)

# Print the dictionaries
for i, data in enumerate(result_dicts, 1):
    print("Dictionary", i, ":", data)
