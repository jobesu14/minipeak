import numpy as np
from pathlib import Path
import  pandas as pd
import pyabf
import torch
from typing import Tuple


def read_abf(abf_file: Path, sampling_rate: int = 1) -> pd.DataFrame:
    abf = pyabf.ABF(abf_file)
    time = abf.sweepX
    amplitude = abf.sweepY
    if sampling_rate > 1:
        time = time[::sampling_rate]
        amplitude = amplitude[::sampling_rate]
    df = pd.DataFrame({'time':time, 'amplitude':amplitude})
    return df


def remove_low_freq_trend(serie_ms: pd.Series, window_ms: int) -> pd.DataFrame: 
     trend = serie_ms.rolling(window = window_ms, center = True).mean()
     return serie_ms - trend


def split_into_overlapping_windows(time_series, window_size: int, overlap: int):
    # Make sure time_series has a length which is a multiple of window_size,
    # if it is not the case fill with zeros
    if len(time_series) % window_size != 0:
        time_series = np.pad(time_series,
                             (0, window_size - len(time_series) % window_size),
                             mode='constant')

    # Split time series into overlapping windows
    n_windows = int((len(time_series) - overlap) / (window_size - overlap))
    windows = np.empty((n_windows, window_size))
    t = 0
    for i in range(n_windows):
        windows[i] = time_series[t:t+window_size]
        t += window_size - overlap
    return windows


def load_experiment_from_csv(csv_file: Path) -> pd.DataFrame:
    # Read csv file into a pandas dataframe
    experiment_df = pd.read_csv(csv_file)
    return experiment_df



def load_windowed_dataset(csv_file: Path, window_size: int = 100) \
        -> Tuple[np.ndarray, np.ndarray]:
    data_df = load_experiment_from_csv(csv_file)

    # Split the data into amplitude (input) and minis (output) variables
    amplitude = data_df['amplitude'].values
    minis = data_df['minis'].astype('bool').values

    amp_win= split_into_overlapping_windows(amplitude, window_size, int(window_size/2))
    minis_win = split_into_overlapping_windows(minis, window_size, int(window_size/2))

    return amp_win, minis_win


def load_training_dataset(folder: Path, window_size: int = 100) \
        -> torch.utils.data.TensorDataset:
    csv_files = folder.glob('*.csv')
    
    all_X = np.empty((0, 1, window_size))
    all_y = np.empty((0, 1, window_size))
    
    for csv_file in csv_files:
        X, y = load_windowed_dataset(csv_file, window_size)
        
        X = X.reshape(-1, 1, window_size)
        y = y.reshape(-1, 1, window_size)
    
        all_X = np.concatenate([all_X, X], axis=0)
        all_y = np.concatenate([all_y, y], axis=0)

    return all_X, all_y
