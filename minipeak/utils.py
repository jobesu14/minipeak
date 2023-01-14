import json
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import  pandas as pd
import pyabf
import torch
from typing import Dict, List, Tuple
import uuid


ABF_SAMPLING_RATE = 100 # we use only 1/100 of the original data
TREND_REMOVAL_WINDOW = 100 # in ms


def remove_low_freq_trend(serie: pd.Series, window: int) -> pd.DataFrame: 
     trend = serie.rolling(window = window, center = True).mean()
     return serie - trend


def group_minis_and_electrophy(minis_df: pd.DataFrame, electrophy_df: pd.DataFrame) -> pd.DataFrame: 
    """ 
    Expand minis_df to match electrophy_df timestep. 
    """
    experiment_df = electrophy_df.copy()
    # remove row with no 'amplitude' data
    experiment_df.dropna(subset=['amplitude'], inplace=True)
    experiment_df['minis'] = 0.0 ## add minis column filled with zeros
    
    mini_id = 0
    t_next_mini = minis_df['time'].iloc[mini_id]
    for index, row in experiment_df.iterrows():
        if mini_id < len(minis_df) - 1 and t_next_mini <= row['time']:
            mini_id += 1
            t_next_mini = minis_df['time'].iloc[mini_id]
            experiment_df.at[index, 'minis'] = row['amplitude']
    return experiment_df


def read_minis(xls_file: Path, exp_name: str) -> pd.DataFrame:
    # create a pandas dataframe with the columns B and G starting at row 5
    df = pd.read_excel(xls_file, sheet_name=exp_name, usecols="B,C,G", skiprows=4,
                       names=['time', 'amplitude', 'baseline'])
    # convert time from ms to s
    df['time'] = df['time'] / 1000.0
    # add amplitude and baseline and set result to amplitude column
    df['amplitude'] = df['amplitude'] + df['baseline']
    # drop baseline column
    df.drop('baseline', axis=1, inplace=True)
    return df


def read_abf(abf_file: Path, sampling_rate: int = 1) -> pd.DataFrame:
    abf = pyabf.ABF(abf_file)
    time = abf.sweepX
    amplitude = abf.sweepY
    if sampling_rate > 1:
        time = time[::sampling_rate]
        amplitude = amplitude[::sampling_rate]
    df = pd.DataFrame({'time':time, 'amplitude':amplitude})
    return df


def load_experiment_from_foder(xls_file: Path, abf_file: Path,
                               exp_name: str, remove_trend: bool) -> pd.DataFrame:
    minis_df = read_minis(xls_file, exp_name)
    electrophy_df = read_abf(abf_file, sampling_rate=ABF_SAMPLING_RATE)
    if remove_trend:
        electrophy_df['amplitude'] = remove_low_freq_trend(electrophy_df['amplitude'],
                                                           window=TREND_REMOVAL_WINDOW)
    experiment_df = group_minis_and_electrophy(minis_df, electrophy_df)
    return experiment_df


def load_experiment_from_csv(csv_file: Path) -> pd.DataFrame:
    # Read csv file into a pandas dataframe
    experiment_df = pd.read_csv(csv_file)
    return experiment_df


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


def convert_minis_to_amplitude(amplitude: np.ndarray, minis: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    peaks_time = []
    peaks_amp = []
    for t, (amp, mini) in enumerate(zip(amplitude, minis)):
        if mini:
            peaks_time.append(t)
            peaks_amp.append(amp)
    return np.array(peaks_time), np.array(peaks_amp)


def load_windowed_dataset(csv_file: Path, window_size: int = 100, overlap: int = 0) \
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
        X, y = load_windowed_dataset(csv_file, window_size, int(window_size/2))
        
        X = X.reshape(-1, 1, window_size)
        y = y.reshape(-1, 1, window_size)
    
        all_X = np.concatenate([all_X, X], axis=0)
        all_y = np.concatenate([all_y, y], axis=0)

    return all_X, all_y


def filter_data_window(all_amplitude: np.ndarray, all_minis: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    # Get same amount of data with and without minis in them.
    win_minis_indices = []
    win_no_minis_indices = []
    for i, mini_win in enumerate(all_minis):
        if np.any(mini_win, axis=1):
            win_minis_indices.append(i)
        else:
            win_no_minis_indices.append(i)

    if len(win_minis_indices) < len(win_no_minis_indices):
        win_no_minis_indices = np.random.choice(win_no_minis_indices,
                                                size=len(win_minis_indices),
                                                replace=False)
    else:
        win_minis_indices = np.random.choice(win_minis_indices,
                                             size=len(win_no_minis_indices),
                                             replace=False)
    indices_to_keep = np.concatenate([win_minis_indices, win_no_minis_indices])

    return np.take(all_amplitude, indices_to_keep, axis=0), \
           np.take(all_minis, indices_to_keep, axis=0)


def create_experiment_folder(root_folder_path: Path) -> Path:
    # create experiment folder if doesn't exist
    experiemnt_folder = root_folder_path / str(uuid.uuid1())
    experiemnt_folder.mkdir(parents=True, exist_ok=True)
    return experiemnt_folder


def save_training_resultsto_csv(experiment_folder: Path, training_df: pd.DataFrame) -> None:
    training_df.to_csv(experiment_folder / 'training_results.csv', index=False)


def save_false_negatives_to_image(experimeent_folder: Path, x: np.ndarray,
                             y_pred: np.ndarray, y: np.ndarray) -> Tuple[int, int]:
    false_neg_path = experimeent_folder / f'false_negatives'
    false_neg_path.mkdir(parents=True, exist_ok=True)
    
    false_neg = 0
    true_neg = 0
    for amplitude, pred, target in zip(x, y_pred, y):
        if pred[0] < 0.5 and target[0] > 0.5:
            false_neg += 1
            image_path = f'{false_neg_path}/{uuid.uuid1()}.png'
            save_window_to_image(image_path, amplitude[0], 'False negatives')
        elif pred[0] < 0.5 and target[0] < 0.5:
            true_neg += 1

    return true_neg, false_neg


def save_false_positives_to_image(experimeent_folder: Path, x: np.ndarray,
                             y_pred: np.ndarray, y: np.ndarray) -> Tuple[int, int]:
    false_pos_path = experimeent_folder / f'false_positives'
    false_pos_path.mkdir(parents=True, exist_ok=True)
    
    false_pos = 0
    true_pos = 0
    for amplitude, pred, target in zip(x, y_pred, y):
        if pred[0] > 0.5 and target[0] < 0.5:
            false_pos += 1
            image_path = f'{false_pos_path}/{uuid.uuid1()}.png'
            save_window_to_image(image_path, amplitude[0], 'False positives')
        elif pred[0] > 0.5 and target[0] > 0.5:
            true_pos += 1

    return true_pos, false_pos


def save_window_to_image(output_file: Path, amplitude: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(amplitude, label='Electrophy')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')
    ax.legend()
    fig.savefig(output_file)


def save_experiment_to_json(experiment_folder: Path, epochs: int, learning_rate: float,
                            weight_decay: float, window_size: int, loss: float,
                            accuracy: float , precision: float, recall: float) -> None:
    exp_dict = {
        'hyperparameters': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'window_size': window_size
        },
        # TODO training data?
        'validation_results': {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    }
    
    with open(experiment_folder / 'experiment.json', 'w') as f:
        json.dump(exp_dict, f, indent=4)