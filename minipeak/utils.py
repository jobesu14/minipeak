import json
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import time
from typing import Optional, Tuple
import uuid


def convert_peaks_to_amplitude(amplitude: np.ndarray, peaks: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Convert peaks boolean mask into peaks amplitude.

    :param amplitude: amplitude data of the timeserie
    :param peaks: boolean mask of the rpesence or absence of a peak at each time of the
    timeserie
    :return: timeserie (time, amplitude) of the peaks
    """
    peaks_time = []
    peaks_amp = []
    for t, (amp, peak) in enumerate(zip(amplitude, peaks)):
        if peak:
            peaks_time.append(t)
            peaks_amp.append(amp)
    return np.array(peaks_time), np.array(peaks_amp)


def peak_percent_to_time_series(peak_percent: float, amplitude: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Compute the timeserie (time, amplitude) of a peak located at 'peak_percent'
    (percentage over the whole 'amplitude' array).

    :param peak_percent: percentage of the peak position over the whole 'amplitude' array
    :param amplitude: amplitude data of the timeserie
    :return: timeserie (time, amplitude) of the peak
    """
    peak_pos_index = int(peak_percent * amplitude.shape[0])
    peaks = np.zeros(amplitude.shape[0], dtype=bool)
    peaks[peak_pos_index] = True
    return convert_peaks_to_amplitude(amplitude, peaks)


def filter_windows(windows_amplitude: np.ndarray, windows_peaks: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Get same amount of data with and without mini peaks in them to balance the learning.

    :param windows_amplitude: windows of amplitude timeseries
    :param windows_peaks: windows of peaks timeseries
    :return: filtered windows of amplitude and peaks timeseries with same amount of
    windows containing peaks and windows without peaks
    """
    win_peaks_indices = []
    win_no_peaks_indices = []
    for i, peak_win in enumerate(windows_peaks):
        if np.any(peak_win, axis=1):
            win_peaks_indices.append(i)
        else:
            win_no_peaks_indices.append(i)

    if len(win_peaks_indices) < len(win_no_peaks_indices):
        win_no_peaks_indices = np.random.choice(win_no_peaks_indices,
                                                size=len(win_peaks_indices),
                                                replace=False)
    else:
        win_peaks_indices = np.random.choice(win_peaks_indices,
                                             size=len(win_no_peaks_indices),
                                             replace=False)
    indices_to_keep = np.concatenate([win_peaks_indices, win_no_peaks_indices])

    return np.take(windows_amplitude, indices_to_keep, axis=0), \
        np.take(windows_peaks, indices_to_keep, axis=0)


def create_experiment_folder(root_folder_path: Path) -> Path:
    """ Create experiment folder based on the date and time if doesn't exist. """
    folder_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    experiment_folder = root_folder_path / folder_name
    experiment_folder.mkdir(parents=True, exist_ok=True)
    return experiment_folder


def save_training_results_to_csv(experiment_folder: Path, training_df: pd.DataFrame) \
        -> None:
    """ Save training progression to csv. """
    training_df.to_csv(experiment_folder / 'training_results.csv', index=False)


def save_false_negatives_to_image(experiment_folder: Path, x: np.ndarray,
                                  y_pred: np.ndarray, y: np.ndarray) -> Tuple[int, int]:
    """
    Save false negatives peak detection plot to image.

    :pram experiment_folder: path to the folder to save the images
    :param x: windows of amplitude timeseries
    :param y_pred: predicted peaks or absence of peak for each window
    :param y: correct currated peaks or absence of peak for each window
    :return: number of true negatives and false negatives
    """
    false_neg_path = experiment_folder / 'false_negatives'
    false_neg_path.mkdir(parents=True, exist_ok=True)

    false_neg = 0
    true_neg = 0
    for amplitude, pred, target in zip(x, y_pred, y):
        if pred[0] < 0.5 and target[0] > 0.5:
            false_neg += 1
            image_path = f'{false_neg_path}/{uuid.uuid1()}.png'
            save_window_to_image(image_path, amplitude[0], None, target[1],
                                 'False negatives')
        elif pred[0] < 0.5 and target[0] < 0.5:
            true_neg += 1

    return true_neg, false_neg


def save_false_positives_to_image(experiment_folder: Path, x: np.ndarray,
                                  y_pred: np.ndarray, y: np.ndarray) -> Tuple[int, int]:
    """
    Save false positives peak detection plot to image.

    :pram experiment_folder: path to the folder to save the images
    :param x: windows of amplitude timeseries
    :param y_pred: predicted peaks or absence of peak for each window
    :param y: correct currated peaks or absence of peak for each window
    :return: number of true positives and false positives
    """
    false_pos_path = experiment_folder / 'false_positives'
    false_pos_path.mkdir(parents=True, exist_ok=True)
    correct_pos_path = experiment_folder / 'correct_positives'
    correct_pos_path.mkdir(parents=True, exist_ok=True)

    false_pos = 0
    true_pos = 0
    for amplitude, pred, target in zip(x, y_pred, y):
        if pred[0] > 0.5 and target[0] < 0.5:
            false_pos += 1
            image_path = f'{false_pos_path}/{uuid.uuid1()}.png'
            save_window_to_image(image_path, amplitude[0], pred[1], None,
                                 'False positives')
        elif pred[0] > 0.5 and target[0] > 0.5:
            true_pos += 1
            image_path = f'{correct_pos_path}/{uuid.uuid1()}.png'
            save_window_to_image(image_path, amplitude[0], pred[1], target[1],
                                 'Correct positives')

    return true_pos, false_pos


def save_window_to_image(output_file: Path, amplitude: np.ndarray,
                         pred_peak_pos_perc: Optional[float],
                         target_peak_pos_perc: Optional[float],
                         title: str) -> None:
    """ Save window of amplitude timeseries with or without peak to image."""
    # Plot data
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('amplitude (mV)')
    ax.plot(amplitude, label='amplitude')

    # Create mini peaks timeseries based on position of the peak
    if pred_peak_pos_perc:
        peaks_time, peaks_amp = \
            peak_percent_to_time_series(pred_peak_pos_perc, amplitude)
        ax.plot(peaks_time, peaks_amp, 'x', label='predicted peak')
    if target_peak_pos_perc:
        peaks_time, peaks_amp = \
            peak_percent_to_time_series(target_peak_pos_perc, amplitude)
        ax.plot(peaks_time, peaks_amp, 'o', label='target peak')

    ax.legend()
    fig.savefig(output_file)
    plt.close(fig)


def save_experiment_to_json(experiment_folder: Path, epochs: int, learning_rate: float,
                            weight_decay: float, window_size: int, loss: float,
                            accuracy: float, pos_error: float, precision: float,
                            recall: float) -> None:
    """ Save experiment hyperparameters and evalutation results to json. """
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
            'pos_error': pos_error,
            'precision': precision,
            'recall': recall
        }
    }

    with open(experiment_folder / 'experiment.json', 'w') as f:
        json.dump(exp_dict, f, indent=4)
