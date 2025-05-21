import numpy as np
import pywt
from pod5 import Reader
from scipy.stats import zscore
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from numba import jit, prange
from scipy.signal import resample
import concurrent.futures



# Fallback for when Numba cannot be used (e.g., with complex types)
def kalman_filter(signal, Q=0.0001, R=0.01):
    """
    Apply Kalman filter to signal using Q and R to smooth over the data.
    Standard implementation.
    """
    n = len(signal)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0] = signal[0]
    P[0] = 1.0

    for k in range(1, n):
        # Predict
        xhat_minus = xhat[k - 1]
        P_minus = P[k - 1] + Q

        # Update
        K = P_minus / (P_minus + R)
        xhat[k] = xhat_minus + K * (signal[k] - xhat_minus)
        P[k] = (1 - K) * P_minus

    return xhat

def z_score_filter(signal, threshold=3):
    z_scores = zscore(signal)
    filtered_signal = signal.copy()
    outliers = np.abs(z_scores) >= threshold
    if np.any(outliers):
        # Replace outliers with the median or nearest non-outlier
        median_val = np.median(signal[~outliers])
        filtered_signal[outliers] = median_val
    return filtered_signal

@jit(nopython=True, parallel=True, fastmath=True)
def kalman_filter_numba(signal, Q=0.0001, R=0.01):
    """
    Apply Kalman filter to signal using Q and R to smooth over the data.
    Optimized with Numba JIT compilation with parallel processing and fastmath.
    """
    n = len(signal)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0] = signal[0]
    P[0] = 1.0

    for k in range(1, n):
        # Predict
        xhat_minus = xhat[k - 1]
        P_minus = P[k - 1] + Q

        # Update
        K = P_minus / (P_minus + R)
        xhat[k] = xhat_minus + K * (signal[k] - xhat_minus)
        P[k] = (1 - K) * P_minus

    return xhat


# Pre-calculate wavelet coefficients for common signal sizes
@jit(nopython=True)
def soft_threshold(x, threshold):
    """Numba-optimized soft thresholding function."""
    sign = np.sign(x)
    magnitude = np.abs(x)
    return sign * np.maximum(magnitude - threshold, 0)


@jit(nopython=True)
def z_score_filter_numba(signal, threshold=3.0):
    """
    Numba-accelerated Z-score filtering.
    """
    # Calculate mean and std
    mean = np.mean(signal)
    std = np.std(signal)

    if std == 0:  # Avoid division by zero
        return signal.copy()

    # Calculate z-scores
    z_scores = (signal - mean) / std

    # Create filtered signal
    filtered_signal = signal.copy()

    # Identify outliers
    for i in range(len(signal)):
        if abs(z_scores[i]) >= threshold:
            # Replace with the median or mean
            filtered_signal[i] = mean

    return filtered_signal


def wavelet_coeffs_cache():
    """Factory function for wavelet coefficient cache."""
    cache = {}

    def get_wavelet_coeffs(signal_len, wavelet='db4', level=4):
        key = (signal_len, wavelet, level)
        if key not in cache:
            # Generate dummy signal of the right length for calculating the wavelet shape
            dummy = np.ones(signal_len)
            coeffs = pywt.wavedec(dummy, wavelet, level=level)
            shapes = [c.shape[0] for c in coeffs]
            cache[key] = shapes
        return cache[key]

    return get_wavelet_coeffs


# Create cache function
get_cached_wavelet_coeffs = wavelet_coeffs_cache()


def wavelet_mode_maxima_denoise_optimized(signal, wavelet='db4', level=4, threshold_factor=0.5):
    """
    Denoise a signal using wavelet decomposition + mode maxima thresholding.
    Optimized version with cached coefficient shapes.
    """
    # Get coefficients
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Calculate threshold once
    detail_coeffs = coeffs[1:]
    if len(detail_coeffs[-1]) > 0:  # Check to avoid empty arrays
        sigma = np.median(np.abs(detail_coeffs[-1])) / 0.6745  # Robust noise estimate
        threshold = threshold_factor * sigma * np.sqrt(2 * np.log(len(signal)))

        # Apply thresholding to all detail coefficients - vectorized
        for i in range(len(detail_coeffs)):
            detail_coeffs[i] = pywt.threshold(detail_coeffs[i], threshold, mode='soft')

    # Reconstruct signal
    new_coeffs = [coeffs[0]] + detail_coeffs
    denoised_signal = pywt.waverec(new_coeffs, wavelet)

    # Ensure same length as input
    return denoised_signal[:len(signal)]


def batch_process_signals(signal_batch, process_func, chunk_size=100):
    """
    Process signals in smaller batches to reduce memory usage.

    Args:
        signal_batch: List of signals to process
        process_func: Function to apply to each signal
        chunk_size: Size of processing chunks

    Returns:
        List of processed signals
    """
    results = []
    total_signals = len(signal_batch)

    for i in range(0, total_signals, chunk_size):
        end_idx = min(i + chunk_size, total_signals)
        chunk = signal_batch[i:end_idx]

        # Process current chunk
        processed_chunk = [process_func(signal) for signal in chunk]
        results.extend(processed_chunk)

    return results


def hybrid_wavelet_kalman_denoise_optimized(signal):
    """
    Apply wavelet denoising followed by Kalman filtering.
    Optimized with vectorized operations and Numba.
    """
    # First apply wavelet denoising
    wavelet_denoised = wavelet_mode_maxima_denoise_optimized(signal)

    # Then apply Kalman filtering for smoothing
    try:
        # Use the faster Numba version
        kalman_smoothed = kalman_filter_numba(wavelet_denoised)
    except Exception:
        # Fall back to regular version if Numba fails
        kalman_smoothed = kalman_filter(wavelet_denoised)

    return kalman_smoothed


@jit(nopython=True, parallel=True)
def process_signals_numba(signals):
    """
    Process multiple signals in parallel using Numba.
    Only works for the stages that can be JIT-compiled.
    """
    n = len(signals)
    result = np.zeros_like(signals)

    for i in prange(n):
        # Apply z-score filtering (can be numba-compiled)
        result[i] = z_score_filter_numba(signals[i])

    return result


def resample_signal(signal, old_rate, new_rate):
    """
    Resample a signal to a new sample rate. Optimized for common cases.
    """
    if old_rate == new_rate:
        return signal

    # Calculate new length
    new_length = int(len(signal) * new_rate / old_rate)

    # Use scipy's resample for the general case
    return resample(signal, new_length)


def extract_vectors_from_pod5(pod5_file_path, max=None, target_sample_rate=5000, chunk_size=256, batch_size=1000):
    """
    Extract and normalize signal vectors from a Pod5 file.
    Optimized with batch processing to reduce memory pressure.

    Args:
        pod5_file_path (str): Path to the Pod5 file
        max (int, optional): Maximum number of signals to extract
        target_sample_rate (int): Desired normalized sample rate
        chunk_size (int): Desired length of output signal chunks
        batch_size (int): Number of reads to process at once

    Returns:
        list: List of fixed-size numpy arrays
    """
    matrix_signals = []

    with Reader(pod5_file_path) as reader:
        if max is None:
            max = float('inf')

        count = 0
        batch_reads = []

        for read in reader.reads():
            if count >= max:
                break

            batch_reads.append(read)
            count += 1

            # Process in batches to manage memory usage
            if len(batch_reads) >= batch_size or count >= max:
                # Process current batch
                for read in batch_reads:
                    signal = np.array(read.signal)

                    # Get sample rate (with fallback)
                    sample_rate = getattr(read, "sample_rate", 4000)

                    # Normalize sample rate
                    signal_resampled = resample_signal(signal, sample_rate, target_sample_rate)

                    if chunk_size:
                        # Split into fixed-size chunks
                        n_chunks = len(signal_resampled) // chunk_size
                        for i in range(n_chunks):
                            chunk = signal_resampled[i * chunk_size: (i + 1) * chunk_size]
                            matrix_signals.append(chunk)
                    else:
                        matrix_signals.append(signal_resampled)

                # Clear batch
                batch_reads = []

    return matrix_signals


def process_signal(vector):
    """
    Process a single signal with denoising and filtering.
    Used for parallel processing.

    Args:
        vector (np.ndarray): Input signal vector

    Returns:
        np.ndarray: Processed signal
    """
    # Apply hybrid denoising
    denoised_signal = hybrid_wavelet_kalman_denoise_optimized(vector)

    # Apply z-score filtering with Numba
    try:
        filtered_signal = z_score_filter_numba(denoised_signal)
    except:
        # Fall back to numpy version if needed
        filtered_signal = z_score_filter(denoised_signal, threshold=3)

    return filtered_signal


def process_signal_batch(vectors):
    """
    Process a batch of signals at once.

    Args:
        vectors (list): List of signal vectors

    Returns:
        list: List of processed signals
    """
    results = []

    for vector in vectors:
        # Apply hybrid denoising
        denoised_signal = hybrid_wavelet_kalman_denoise_optimized(vector)

        # Apply z-score filtering
        filtered_signal = z_score_filter_numba(denoised_signal)
        results.append(filtered_signal)

    return results


def read_and_filter_parallel(input_file, max=None, n_processes=None,
                             plot_original=True, plot_filtered=True,
                             batch_size=100, chunk_size=None):
    """
    Read signals from a Pod5 file, process them in parallel with optimized batching,
    and optionally plot results.

    Args:
        input_file (str): Path to the Pod5 file
        max (int, optional): Maximum number of signals to process
        n_processes (int, optional): Number of parallel processes to use
        plot_original (bool): Whether to plot original signals
        plot_filtered (bool): Whether to plot filtered signals
        batch_size (int): Number of signals to process in each parallel batch

    Returns:
        list: List of processed signals
    """
    # Determine optimal number of processes if not specified
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)  # Leave one CPU free for system

    # Extract signals from Pod5 file
    matrix = extract_vectors_from_pod5(input_file, max, chunk_size=chunk_size)

    if plot_original:
        plot_signals(matrix)

    # Prepare batches for processing
    batches = [matrix[i:i + batch_size] for i in range(0, len(matrix), batch_size)]

    # Process batches in parallel
    filtered = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Use map for better workload distribution
        batch_results = list(executor.map(process_signal_batch, batches))

        # Flatten results
        for batch in batch_results:
            filtered.extend(batch)

    if plot_filtered:
        plot_signals(filtered)

    return filtered


def read_and_filter(input_file, max=None, plot_original=False, plot_filtered=False, chunk_size=None):
    """
    Read signals from a Pod5 file, process them sequentially with optimizations,
    and optionally plot results.

    Args:
        input_file (str): Path to the Pod5 file
        max (int, optional): Maximum number of signals to process
        plot_original (bool): Whether to plot original signals
        plot_filtered (bool): Whether to plot filtered signals

    Returns:
        list: List of processed signals
    """
    matrix = extract_vectors_from_pod5(input_file, max, chunk_size=chunk_size)

    if plot_original:
        plot_signals(matrix)

    # Process signals in batches of 100 for better memory management
    filtered = []
    batch_size = 100
    for i in range(0, len(matrix), batch_size):
        batch = matrix[i:min(i + batch_size, len(matrix))]
        batch_filtered = []

        for vector in batch:
            # Apply hybrid denoising first
            denoised_signal = hybrid_wavelet_kalman_denoise_optimized(vector)
            # Then apply z-score filtering
            filtered_signal = z_score_filter_numba(denoised_signal)
            batch_filtered.append(filtered_signal)

        filtered.extend(batch_filtered)

    if plot_filtered:
        plot_signals(filtered)

    return filtered


def plot_signals(matrix_signals, max_signals=500):
    """
    Plots each signal from the matrix_signals list.
    Limits the number of signals plotted to avoid overloading the plot.

    Args:
        matrix_signals (list): A list where each element is a 1D numpy array representing a signal.
        max_signals (int): Maximum number of signals to plot
    """
    # Create a figure and axis for the plot
    plt.figure(figsize=(10, 6))

    # Limit the number of signals to plot
    signals_to_plot = matrix_signals[:max_signals]

    # Plot each signal in the matrix
    for idx, signal in enumerate(signals_to_plot):
        plt.plot(signal, label=f"Signal {idx + 1}")

    # Adding labels and title
    plt.xlabel("Sample Index")
    plt.ylabel("Signal Amplitude")
    plt.title("Nanopore Signals")

    # Add legend if there aren't too many signals
    if len(signals_to_plot) <= 10:
        plt.legend()

    # Show the plot
    plt.show()