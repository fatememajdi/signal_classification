import numpy as np
import pandas as pd
import os
from glob import glob
from sklearn.preprocessing import MinMaxScaler
import re
from scipy.ndimage import zoom
from scipy.signal import spectrogram


def load_signal_and_fs(path):
    """
    Load a vibration signal and compute its sampling frequency from a .txt file.

    The function reads header metadata (such as Max_X and NoOfItems) to extract
    the total duration and number of samples, and uses them to compute the sampling rate.
    It also loads the signal values, ignoring comment lines starting with '%'.

    Args:
        path (str): Path to the .txt file containing the signal and metadata.

    Returns:
        tuple:
            - signal (np.ndarray): The loaded 1D signal array.
            - fs (float): The computed sampling frequency (samples per second).

    Raises:
        ValueError: If the necessary metadata (duration or number of samples) cannot be extracted.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    duration = None
    N = None
    for line in lines:
        if line.startswith('%'):
            if 'Max_X' in line:
                match = re.search(r'Max_X:\s*=\s*([0-9.]+)', line)
                if match:
                    duration = float(match.group(1))
            elif 'NoOfItems' in line:
                match = re.search(r'NoOfItems:\s*=\s*([0-9]+)', line)
                if match:
                    N = int(match.group(1))
        else:
            break

    if duration is None or N is None:
        raise ValueError(
            "Failed to extract duration or number of samples from header")

    fs = N / duration
    signal = np.loadtxt(path, comments='%')

    return signal, fs


def signal_to_spectrogram(signal, fs, nperseg=None, noverlap=None, target_shape=(128, 128)):
    """
    Convert a 1D time-domain signal to a 2D spectrogram image.

    Applies Short-Time Fourier Transform (STFT) to extract time-frequency features.
    The spectrogram is then log-scaled, normalized to [0, 1], and resized to a fixed shape.

    Args:
        signal (np.ndarray): Input 1D signal array.
        fs (float): Sampling frequency of the signal.
        nperseg (int, optional): Window size for STFT. If None, defaults to fs / 5.
        noverlap (int, optional): Overlap between windows. Defaults to nperseg / 3.
        target_shape (tuple): Desired output size for the spectrogram (height, width).

    Returns:
        np.ndarray: 2D normalized and resized spectrogram image.
    """
    if nperseg is None:
        nperseg = int(fs / 5)
    if noverlap is None:
        noverlap = int(nperseg / 3)

    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx_log = np.log(Sxx + 1e-6)
    Sxx_min = np.min(Sxx_log)
    Sxx_max = np.max(Sxx_log)
    Sxx_norm = (Sxx_log - Sxx_min) / (Sxx_max - Sxx_min + 1e-10)
    Sxx_resized = zoom(
        Sxx_norm, (target_shape[0] / Sxx_norm.shape[0], target_shape[1] / Sxx_norm.shape[1]))
    return Sxx_resized


def load_dataset(healthy_dir, faulty_dir, output_pkl_path):
    """
    Load and preprocess all signals from given directories and convert them to spectrograms.

    For each .txt signal file, this function:
    - Loads the signal and computes sampling rate
    - Converts the signal to a normalized spectrogram
    - Labels it as healthy (0) or faulty (1)

    Args:
        healthy_dir (str): Directory path containing healthy signal files.
        faulty_dir (str): Directory path containing faulty signal files.

    Returns:
        pd.DataFrame: DataFrame containing the following columns:
            - 'filename': Name of the signal file
            - 'label': 0 for healthy, 1 for faulty
            - 'fs': Sampling frequency
            - 'signal': Original 1D signal
            - 'spectrogram': 2D spectrogram representation
    """
    data = []

    for label, directory in [(0, healthy_dir), (1, faulty_dir)]:
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".txt"):
                path = os.path.join(directory, filename)
                signal, fs = load_signal_and_fs(path)
                spec = signal_to_spectrogram(signal, fs=fs)
                data.append({
                    'filename': filename,
                    'label': label,
                    'fs': fs,
                    'signal': signal,
                    'spectrogram': spec
                })
    
    df = pd.DataFrame(data)

    if output_pkl_path:
        df.to_pickle(output_pkl_path)
    return df


def load_signal(filepath):
    """
    Load a time-domain signal from a text file.

    This function reads a .txt file containing signal values, ignores metadata 
    or comment lines that start with '%' and returns the signal as a NumPy array.

    Args:
        filepath (str): Path to the .txt file containing the signal.

    Returns:
        np.ndarray: 1D array of float values representing the signal.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data_lines = [line.strip() for line in lines if not line.startswith(
        '%') and line.strip() != '']
    signal = np.array([float(x) for x in data_lines])
    return signal


def load_all_signals(healthy_dir, faulty_dir, output_pkl_path):
    """
    Load and label all signal files from healthy and faulty directories.

    This function reads all `.txt` files in the provided directories,
    assigns label 0 to healthy signals and 1 to faulty ones,
    and stores them in a pandas DataFrame along with filenames.
    Optionally, the resulting DataFrame can be saved as a `.pkl` file.

    Args:
        healthy_dir (str): Path to directory containing healthy signals.
        faulty_dir (str): Path to directory containing faulty signals.
        output_pkl_path (str): Path to save the resulting DataFrame as a pickle file (.pkl).
                               If None or empty, the file won't be saved.

    Returns:
        pd.DataFrame: DataFrame containing signal data, labels, and filenames.
    """
    data = []

    for path in glob(os.path.join(healthy_dir, '*.txt')):
        signal = load_signal(path)
        data.append({'signal': signal, 'label': 0,
                     'filename': os.path.basename(path)})

    for path in glob(os.path.join(faulty_dir, '*.txt')):
        signal = load_signal(path)
        data.append({'signal': signal, 'label': 1,
                     'filename': os.path.basename(path)})

    df = pd.DataFrame(data)

    if output_pkl_path:
        df.to_pickle(output_pkl_path)

    return df


def fix_signal_length(signal, desired_length=25000):
    """
    Adjust the length of a 1D signal to a fixed size by trimming or padding.

    If the signal is longer than `desired_length`, it will be center-cropped.
    If the signal is shorter, it will be zero-padded at the end.
    If it's already the correct length, it is returned unchanged.

    Args:
        signal (np.ndarray): Input 1D signal array.
        desired_length (int): Target length of the signal.

    Returns:
        np.ndarray: Signal of shape (desired_length,), trimmed or padded as needed.
    """
    current_length = len(signal)
    if current_length > desired_length:
        start = (current_length - desired_length) // 2
        return signal[start:start+desired_length]
    elif current_length < desired_length:
        pad_width = desired_length - current_length
        return np.pad(signal, (0, pad_width), mode='constant')
    else:
        return signal


def normalize_signal(signal):
    """
    Normalize a 1D signal using Min-Max scaling to the [0, 1] range.

    This function reshapes the input signal to 2D, applies sklearn's MinMaxScaler,
    and flattens it back to 1D after scaling.

    Args:
        signal (np.ndarray): 1D NumPy array representing the raw signal.

    Returns:
        np.ndarray: Normalized signal with values scaled between 0 and 1.
    """
    scaler = MinMaxScaler()
    norm_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    return norm_signal
