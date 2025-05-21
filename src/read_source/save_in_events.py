from .pod5_reader import read_and_filter
import os
import numpy as np


def uniform_quantize(signal: np.ndarray, num_levels: int = 64):
    """
    Uniformly quantizes a signal into `num_levels` bins.
    Parameters:
        signal (np.ndarray): 1D array of signal values (float).
        num_levels (int): Number of quantization levels.
    Returns:
        np.ndarray: Quantized signal as integer levels (0 to num_levels - 1).
    """
    min_val = np.min(signal)
    max_val = np.max(signal)

    if max_val == min_val:
        return np.zeros_like(signal, dtype=int)

    scaled = (signal - min_val) / (max_val - min_val)
    quantized = np.floor(scaled * num_levels).astype(int)

    quantized = np.clip(quantized, 0, num_levels - 1)
    return quantized


def fold_repeats(signal: np.ndarray) -> np.ndarray:
    """
    Removes consecutive duplicates from a 1D array.
    Parameters:
        signal (np.ndarray): Input array with possible consecutive repeated values.
    Returns:
        np.ndarray: Array with consecutive duplicates removed.
    """
    if signal.size == 0:
        return signal
    return signal[np.insert(signal[1:] != signal[:-1], 0, True)]


def save_in_events(input_dir):
    directory = os.fsencode(input_dir)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".pod5"):
            read_file = os.path.join(input_dir, filename)
            print(f"reading {read_file}")

            # Read and process the file
            temp = read_and_filter(read_file, max=500, chunk_size=1000)

            processed_vectors = []

            for vec in temp:
                quant = uniform_quantize(vec)
                folded_quant = fold_repeats(quant)
                processed_vectors.append(folded_quant)

            save_path = os.path.join(input_dir, f"{os.path.splitext(filename)[0]}.npz")
            np.savez_compressed(save_path, *processed_vectors)
            print(f"saved processed data to {save_path}")