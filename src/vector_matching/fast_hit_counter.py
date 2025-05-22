from numba import jit, prange
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tqdm
from functools import lru_cache


@jit(nopython=True, fastmath=True)
def longest_consecutive_match(vec1, vec2):
    """Optimized function to find longest consecutive matching elements using Numba."""
    max_len = 0
    len1, len2 = len(vec1), len(vec2)

    # Early exit for empty vectors
    if len1 == 0 or len2 == 0:
        return 0

    # Use a 2D array for Numba compatibility (dictionaries aren't supported in nopython mode)
    dp = np.zeros((len1, len2), dtype=np.int32)

    for i in range(len1):
        for j in range(len2):
            if vec1[i] == vec2[j]:
                # Get the value of the previous consecutive match + 1
                if i > 0 and j > 0:
                    dp[i, j] = dp[i - 1, j - 1] + 1
                else:
                    dp[i, j] = 1

                max_len = max(max_len, dp[i, j])

    return max_len


# Optional: Add a parallel version of longest_consecutive_match for large vectors
@jit(nopython=True, parallel=True, fastmath=True)
def longest_consecutive_match_parallel(vec1, vec2):
    """Parallel version of longest_consecutive_match for large vectors."""
    max_len = 0
    len1, len2 = len(vec1), len(vec2)

    # Early exit for empty vectors
    if len1 == 0 or len2 == 0:
        return 0

    # For very small vectors, use the non-parallel version
    if len1 * len2 < 10000:
        return longest_consecutive_match(vec1, vec2)

    # Use an array for results from parallel operations
    results = np.zeros(len1, dtype=np.int32)

    # Process each row in parallel
    for i in prange(len1):
        row_max = 0
        for j in range(len2):
            if vec1[i] == vec2[j]:
                current = 1
                # Check for previous match
                if i > 0 and j > 0:
                    for k in range(1, min(i + 1, j + 1)):
                        if i - k < 0 or j - k < 0 or vec1[i - k] != vec2[j - k]:
                            break
                        current += 1
                row_max = max(row_max, current)
        results[i] = row_max

    return np.max(results)


@jit(nopython=True, fastmath=True)
def longest_consecutive_match_with_tolerance(vec1, vec2, max_tolerance=1):
    """Numba accelerated function to find longest consecutive matching elements with tolerance."""
    if len(vec1) == 0 or len(vec2) == 0:
        return 0, 0

    max_len = 0
    max_tolerated = 0
    len1, len2 = len(vec1), len(vec2)

    # Numba-compatible implementation
    for start1 in range(len1):
        for start2 in range(len2):
            length = 0
            tolerated = 0
            max_possible = min(len1 - start1, len2 - start2)

            # Early stopping if we can't beat current max_len
            if max_possible <= max_len:
                continue

            for offset in range(max_possible):
                if vec1[start1 + offset] != vec2[start2 + offset]:
                    tolerated += 1
                    if tolerated > max_tolerance:
                        break
                length += 1

            # Update best result if better
            if length > max_len or (length == max_len and tolerated < max_tolerated):
                max_len = length
                max_tolerated = tolerated

    return max_len, max_tolerated


# Optional: Add a parallel version of longest_consecutive_match_with_tolerance for large vectors
@jit(nopython=True, parallel=True, fastmath=True)
def longest_consecutive_match_with_tolerance_parallel(vec1, vec2, max_tolerance=1):
    """Parallel version of longest_consecutive_match_with_tolerance for large vectors."""
    if len(vec1) == 0 or len(vec2) == 0:
        return 0, 0

    len1, len2 = len(vec1), len(vec2)

    # For very small vectors, use the non-parallel version
    if len1 * len2 < 10000:
        return longest_consecutive_match_with_tolerance(vec1, vec2, max_tolerance)

    # Use arrays for results from parallel operations
    results_len = np.zeros(len1, dtype=np.int32)
    results_tol = np.zeros(len1, dtype=np.int32)

    # Process each starting position in parallel
    for start1 in prange(len1):
        local_max_len = 0
        local_max_tolerated = 0

        for start2 in range(len2):
            length = 0
            tolerated = 0
            max_possible = min(len1 - start1, len2 - start2)

            # Early stopping if we can't beat current max_len
            if max_possible <= local_max_len:
                continue

            for offset in range(max_possible):
                if start1 + offset >= len1 or start2 + offset >= len2:
                    break

                if vec1[start1 + offset] != vec2[start2 + offset]:
                    tolerated += 1
                    if tolerated > max_tolerance:
                        break
                length += 1

            # Update local best
            if length > local_max_len or (length == local_max_len and tolerated < local_max_tolerated):
                local_max_len = length
                local_max_tolerated = tolerated

        results_len[start1] = local_max_len
        results_tol[start1] = local_max_tolerated

    # Find the max length
    max_len_idx = np.argmax(results_len)
    max_len = results_len[max_len_idx]
    max_tolerated = results_tol[max_len_idx]

    return max_len, max_tolerated


# Helper function to automatically choose the best implementation based on input size
def smart_longest_consecutive_match(vec1, vec2):
    """Automatically choose between parallel and serial implementation based on input size."""
    # For large vectors, use parallel implementation
    if len(vec1) * len(vec2) > 100000:
        return longest_consecutive_match_parallel(vec1, vec2)
    else:
        return longest_consecutive_match(vec1, vec2)


# Helper function to automatically choose the best implementation with tolerance based on input size
def smart_longest_consecutive_match_with_tolerance(vec1, vec2, max_tolerance=1):
    """Automatically choose between parallel and serial implementation with tolerance based on input size."""
    # For large vectors, use parallel implementation
    if len(vec1) * len(vec2) > 100000:
        return longest_consecutive_match_with_tolerance_parallel(vec1, vec2, max_tolerance)
    else:
        return longest_consecutive_match_with_tolerance(vec1, vec2, max_tolerance)


# Wrap the function with LRU cache
@lru_cache(maxsize=1024)
def cached_match(vec1_tuple, vec2_tuple, use_tolerance=False, max_tolerance=1):
    """Cached version of matching functions using hashable tuples."""
    vec1 = np.array(vec1_tuple)
    vec2 = np.array(vec2_tuple)

    if use_tolerance:
        result = smart_longest_consecutive_match_with_tolerance(vec1, vec2, max_tolerance)
        # Return only the match length for consistency
        return result[0]
    else:
        return smart_longest_consecutive_match(vec1, vec2)


def save_in_hits(input_dir, vector, use_tolerance=False, max_tolerance=1):
    """
    Find file with single best matching vector.

    Parameters:
    -----------
    input_dir : str
        Directory containing .npz files
    vector : array-like
        Vector to match against
    use_tolerance : bool
        If True, use longest_consecutive_match_with_tolerance, otherwise use longest_consecutive_match
    max_tolerance : int
        Maximum number of mismatches allowed when use_tolerance is True
    """
    directory = os.fsencode(input_dir)
    file_list = [os.fsdecode(file) for file in os.listdir(directory) if os.fsdecode(file).endswith(".npz")]

    best_file = None
    best_match_length = 0
    best_hit_dict = {}
    all_hit_dicts = {}

    match_mode = "With Tolerance" if use_tolerance else "Exact"
    print(f"Using match mode: {match_mode}" + (f", max_tolerance={max_tolerance}" if use_tolerance else ""))

    vector = np.asarray(vector)
    vector_tuple = tuple(vector)

    for filename in tqdm.tqdm(file_list, desc="Processing files", unit="file"):
        filepath = os.path.join(input_dir, filename)

        try:
            data = np.load(filepath, allow_pickle=True, mmap_mode='r')
            hit_dict = {}
            file_best_match = 0

            for key in data.files:
                vec = np.asarray(data[key])
                vec_tuple = tuple(vec)
                match_len = cached_match(vector_tuple, vec_tuple, use_tolerance, max_tolerance)

                if match_len > 0:
                    hit_dict[key] = match_len
                    if match_len > file_best_match:
                        file_best_match = match_len

            all_hit_dicts[filename] = hit_dict

            if file_best_match > best_match_length:
                best_file = filename
                best_match_length = file_best_match
                best_hit_dict = hit_dict

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return best_file, best_hit_dict, all_hit_dicts


def save_in_hits_average(input_dir, vector, use_tolerance=False, max_tolerance=1):
    """
    Find file with highest average match length (for non-zero matches).

    Parameters:
    -----------
    input_dir : str
        Directory containing .npz files
    vector : array-like
        Vector to match against
    use_tolerance : bool
        If True, use longest_consecutive_match_with_tolerance, otherwise use longest_consecutive_match
    max_tolerance : int
        Maximum number of mismatches allowed when use_tolerance is True
    """
    directory = os.fsencode(input_dir)
    file_list = [os.fsdecode(file) for file in os.listdir(directory) if os.fsdecode(file).endswith(".npz")]

    best_file = None
    best_avg_match = 0
    best_hit_dict = {}
    all_hit_dicts = {}

    match_mode = "With Tolerance" if use_tolerance else "Exact"
    print(f"Using match mode: {match_mode}" + (f", max_tolerance={max_tolerance}" if use_tolerance else ""))

    # Convert input vector to numpy array and tuple
    vector = np.asarray(vector)
    vector_tuple = tuple(vector)

    for filename in tqdm.tqdm(file_list, desc="Processing files", unit="file"):
        filepath = os.path.join(input_dir, filename)

        try:
            data = np.load(filepath, allow_pickle=True, mmap_mode='r')
            hit_dict = {}
            match_lengths = []

            for key in data.files:
                vec = np.asarray(data[key])
                vec_tuple = tuple(vec)
                match_len = cached_match(vector_tuple, vec_tuple, use_tolerance, max_tolerance)

                if match_len > 0:
                    hit_dict[key] = match_len
                    match_lengths.append(match_len)

            all_hit_dicts[filename] = hit_dict
            # Compute average only over non-zero matches
            if match_lengths:
                avg_match = sum(match_lengths) / len(match_lengths)

                if avg_match > best_avg_match:
                    best_avg_match = avg_match
                    best_file = filename
                    best_hit_dict = hit_dict

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return best_file, best_hit_dict, all_hit_dicts


def plot_match_distributions(all_hit_dicts, save_path='match_distributions.png'):
    """
    Plot and save normal distributions of match lengths from all .npz files.
    """
    plt.figure(figsize=(12, 7))

    for filename, hit_dict in all_hit_dicts.items():
        match_lengths = list(hit_dict.values())
        if len(match_lengths) < 2:
            continue  # Skip files with too few matches to estimate std

        mu, std = np.mean(match_lengths), np.std(match_lengths)

        # Create range for PDF
        x = np.linspace(min(match_lengths), max(match_lengths), 100)
        pdf = norm.pdf(x, mu, std)

        plt.plot(x, pdf, label=filename)

    plt.title("Normal Distributions of Match Lengths per .npz File")
    plt.xlabel("Match Length")
    plt.ylabel("Probability Density")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[✓] Distribution plot saved to: {save_path}")