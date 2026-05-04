import os
import glob
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
import json


def reconstruct_sparse_square_matrix(input_file, matrix_output_file, mapping_output_file):
    """
    Reconstructs a sparse square connectivity matrix from taxi CSV file(s).

    Parameters:
        input_file (str): Path to a CSV file or a folder containing CSV files.
        matrix_output_file (str): Path to save the resulting sparse matrix in .npz format.
        mapping_output_file (str): Path to save the location ID to matrix index mapping in JSON format.
    """

    # Read one CSV or all CSVs in a folder
    if os.path.isdir(input_file):
        csv_files = sorted(glob.glob(os.path.join(input_file, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in folder: {input_file}")

        print(f"Found {len(csv_files)} CSV files in folder: {input_file}")
        df_list = []
        for file in csv_files:
            print(f"Loading {file} ...")
            df_list.append(pd.read_csv(file))
        df = pd.concat(df_list, ignore_index=True)
    else:
        print(f"Loading {input_file} ...")
        df = pd.read_csv(input_file)

    # Keep only necessary columns and drop missing rows
    df = df[['pulocationid', 'dolocationid']].dropna()

    # Convert to int
    df['pulocationid'] = df['pulocationid'].astype(int)
    df['dolocationid'] = df['dolocationid'].astype(int)

    # Get all unique location IDs
    unique_ids = pd.unique(df[['pulocationid', 'dolocationid']].values.ravel())

    # Map location IDs to matrix indices
    id_to_index = {int(id_val): idx for idx, id_val in enumerate(unique_ids)}

    # Count trips for each (PU, DO) pair
    edge_counts = (
        df.groupby(['pulocationid', 'dolocationid'])
          .size()
          .reset_index(name='count')
    )

    # Convert to matrix indices
    row_indices = edge_counts['pulocationid'].map(id_to_index).to_numpy()
    col_indices = edge_counts['dolocationid'].map(id_to_index).to_numpy()
    data = edge_counts['count'].to_numpy(dtype=np.int64)

    # Create sparse square count connectivity matrix
    sparse_matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(unique_ids), len(unique_ids))
    )

    # Save the sparse matrix
    save_npz(matrix_output_file, sparse_matrix)
    print(f"Sparse connectivity matrix saved to {matrix_output_file}")
    print(f"Matrix shape: {sparse_matrix.shape}")
    print(f"Number of nonzero entries: {sparse_matrix.nnz}")

    # Save the location ID to matrix index mapping
    json_compatible_mapping = {str(key): value for key, value in id_to_index.items()}
    with open(mapping_output_file, 'w') as f:
        json.dump(json_compatible_mapping, f)
    print(f"Location ID to matrix index mapping saved to {mapping_output_file}")


if __name__ == "__main__":
    # Set this to one CSV file or the whole folder
    input_csv_file = "nyc-taxi-trip"
    matrix_output_file = "sparse_connectivity_matrix.npz"
    mapping_output_file = "location_id_to_index_mapping.json"

    reconstruct_sparse_square_matrix(input_csv_file, matrix_output_file, mapping_output_file)
