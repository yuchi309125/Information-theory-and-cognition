import json

def load_mapping(mapping_file):
    """
    Load the root ID to matrix index mapping from a JSON file.
    
    Parameters:
        mapping_file (str): Path to the JSON file containing the mapping.
    
    Returns:
        dict: A dictionary mapping matrix indices to root IDs.
    """
    with open(mapping_file, 'r') as f:
        id_to_index = json.load(f)
    
    # Convert keys back to integers for use in reverse mapping
    index_to_id = {int(index): int(root_id) for root_id, index in id_to_index.items()}
    return index_to_id

def matrix_index_to_root_id(index, mapping):
    """
    Convert a matrix index to the corresponding root ID.
    
    Parameters:
        index (int): The matrix index to convert.
        mapping (dict): A dictionary mapping matrix indices to root IDs.
    
    Returns:
        int: The corresponding root ID, or None if the index is not found.
    """
    return mapping.get(index)

if __name__ == "__main__":
    # Path to the mapping JSON file
    mapping_file = "root_id_to_index_mapping.json"
    
    # Load the mapping
    index_to_id_mapping = load_mapping(mapping_file)
    
    # Test conversion from matrix index to root ID
    test_index = 0  # Replace with the matrix index you want to test
    root_id = matrix_index_to_root_id(test_index, index_to_id_mapping)
    
    if root_id is not None:
        print(f"Matrix index {test_index} corresponds to root ID {root_id}.")
    else:
        print(f"Matrix index {test_index} not found in the mapping.")