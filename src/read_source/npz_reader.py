import numpy as np

def get_vector_by_index(npz_path, index):
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())

    if index < 0 or index >= len(keys):
        raise IndexError(f"Index {index} out of range. File contains {len(keys)} vectors.")

    key = keys[index]
    return data[key]
