import pandas as pd
import pickle

# Define file paths
input_file = "visual_neuron_types.csv.gz"
output_file = "root_id_type_dict.pkl"

def create_root_id_type_dict(input_file, output_file):
    # Load the gzipped CSV file
    print("Loading CSV file...")
    df = pd.read_csv(input_file)

    # Create the dictionary (root_id, type)
    print("Creating dictionary...")
    root_id_type_dict = dict(zip(df['root_id'], df['type']))

    # Save the dictionary as a pickle file
    print("Saving dictionary as pickle file...")
    with open(output_file, 'wb') as f:
        pickle.dump(root_id_type_dict, f)

    print(f"Dictionary saved to {output_file}")

if __name__ == "__main__":
    create_root_id_type_dict(input_file, output_file)
