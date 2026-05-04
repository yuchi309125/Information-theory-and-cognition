import gzip
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
import json

def reconstruct_sparse_square_matrix(input_file, matrix_output_file, mapping_output_file):
    """
    Reconstructs a sparse square connectivity matrix {0,1} from a gzipped CSV file.

    Parameters:
        input_file (str): Path to the gzipped CSV file containing the edges.
        matrix_output_file (str): Path to save the resulting sparse matrix in .npz format.
        mapping_output_file (str): Path to save the root ID to matrix index mapping in JSON format.
    """
    # Read the gzipped CSV file into a DataFrame
    with gzip.open(input_file, 'rt') as f:
        df = pd.read_csv(f)
    
    # Get all unique root IDs (shared between pre_root_id and post_root_id)
    unique_ids = pd.unique(df[['pre_root_id', 'post_root_id']].values.ravel())
    
    # Map root IDs to matrix indices
    id_to_index = {int(id_val): idx for idx, id_val in enumerate(unique_ids)}
    
    # Convert pre_root_id and post_root_id to matrix indices
    row_indices = df['pre_root_id'].map(id_to_index)
    col_indices = df['post_root_id'].map(id_to_index)
    
    # Create a sparse square binary connectivity matrix
    data = np.ones(len(df), dtype=np.int8)  # All edges are set to 1
    sparse_matrix = csr_matrix((data, (row_indices, col_indices)), 
                               shape=(len(unique_ids), len(unique_ids)))
    
    # Save the sparse matrix to a .npz file
    save_npz(matrix_output_file, sparse_matrix)
    print(f"Sparse connectivity matrix saved to {matrix_output_file}")
    
    # Save the root ID to matrix index mapping to a JSON file
    json_compatible_mapping = {str(key): value for key, value in id_to_index.items()}
    with open(mapping_output_file, 'w') as f:
        json.dump(json_compatible_mapping, f)
    print(f"Root ID to matrix index mapping saved to {mapping_output_file}")

if __name__ == "__main__":
    # Input gzipped CSV file, output files for matrix and mapping
    input_csv_file = "connections.csv.gz"
    matrix_output_file = "sparse_connectivity_matrix.npz"
    mapping_output_file = "root_id_to_index_mapping.json"
    
    # Run the reconstruction process
    reconstruct_sparse_square_matrix(input_csv_file, matrix_output_file, mapping_output_file)