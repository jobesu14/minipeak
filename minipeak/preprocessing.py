import numpy as np
from pathlib import Path
import pandas as pd
import pyabf
from typing import Tuple


def read_abf(abf_file: Path, sampling_rate: int = 1) -> pd.DataFrame:
    """
    Read the electrophysiology data from the abf file and return a dataframe with
    time and amplitude columns. If sampling_rate is greater than 1, the electrophysiology
    data will be downsampled by the given sampling rate.

    :param abf_file (Path): path to the abf file for this experiment
    :param sampling_rate (int): sampling rate to use for the electrophysiology data
    :return: dataframe with electrophysiology amplitude and time
    """
    abf = pyabf.ABF(abf_file)
    time = abf.sweepX
    amplitude = abf.sweepY
    if sampling_rate > 1:
        time = time[::sampling_rate]
        amplitude = amplitude[::sampling_rate]
    df = pd.DataFrame({'time': time, 'amplitude': amplitude})
    return df


def remove_low_freq_trend(serie_ms: pd.Series, window_ms: int) -> pd.DataFrame:
    """ Remove the low frequency trend from the timeserie. """
    trend = serie_ms.rolling(window=window_ms, center=True).mean()
    return serie_ms - trend


def split_into_overlapping_windows(time_series: np.ndarray, window_size: int,
                                   overlap: int) -> np.ndarray:
    """ Split the time series into overlapping windows of size window_size. """
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


def load_training_data_from_csv(csv_file: Path) -> pd.DataFrame:
    """ Load the triannig data from the csv file. """
    experiment_df = pd.read_csv(csv_file)
    return experiment_df


def load_windowed_dataset(csv_file: Path, window_size: int = 100) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the training data from the csv file and split it into overlapping windows of
    size window_size.

    :param csv_file (Path): path to the csv file containing the training data
    :param window_size (int): size of the windows to use for the training data
    :return: tuple of numpy arrays containing the amplitude and minis windows
    """
    data_df = load_training_data_from_csv(csv_file)

    # Split the data into amplitude (input) and minis (output) variables
    amplitude = data_df['amplitude'].values
    minis = data_df['minis'].astype('bool').values

    amp_win = split_into_overlapping_windows(amplitude, window_size, int(window_size/2))
    minis_win = split_into_overlapping_windows(minis, window_size, int(window_size/2))

    return amp_win, minis_win


def load_training_datasets(folder: Path, window_size: int = 100) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and concatenate all the training datasets from the training data csv file in the
    given folder.

     :param folder (Path): path to the folder containing the training data csv files
     :param window_size (int): size of the windows to use for the training data
     :return: tuple of numpy arrays containing the amplitude and minis windowed datasets
    """
    csv_files = folder.glob('*.csv')

    all_x = np.empty((0, 1, window_size))
    all_y = np.empty((0, 1, window_size))

    for csv_file in csv_files:
        x, y = load_windowed_dataset(csv_file, window_size)

        x = x.reshape(-1, 1, window_size)
        y = y.reshape(-1, 1, window_size)

        all_x = np.concatenate([all_x, x], axis=0)
        all_y = np.concatenate([all_y, y], axis=0)

    return all_x, all_y
