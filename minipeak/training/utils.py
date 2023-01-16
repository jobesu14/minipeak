import json
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import  pandas as pd
import time
from typing import Optional, Tuple
import uuid


def convert_minis_to_amplitude(amplitude: np.ndarray, minis: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    peaks_time = []
    peaks_amp = []
    for t, (amp, mini) in enumerate(zip(amplitude, minis)):
        if mini:
            peaks_time.append(t)
            peaks_amp.append(amp)
    return np.array(peaks_time), np.array(peaks_amp)


def peak_percent_to_time_series(peak_percent: float, amplitude: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    peak_pos_index = int(peak_percent * amplitude.shape[0])
    minis = np.zeros(amplitude.shape[0], dtype=bool)
    minis[peak_pos_index] = True
    return convert_minis_to_amplitude(amplitude, minis)


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
    folder_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    experiment_folder = root_folder_path / folder_name
    experiment_folder.mkdir(parents=True, exist_ok=True)
    return experiment_folder


def save_training_resultsto_csv(experiment_folder: Path, training_df: pd.DataFrame) \
        -> None:
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
            save_window_to_image(image_path, amplitude[0], None, target[1],
                                 'False negatives')
        elif pred[0] < 0.5 and target[0] < 0.5:
            true_neg += 1

    return true_neg, false_neg


def save_false_positives_to_image(experimeent_folder: Path, x: np.ndarray,
                             y_pred: np.ndarray, y: np.ndarray) -> Tuple[int, int]:
    false_pos_path = experimeent_folder / f'false_positives'
    false_pos_path.mkdir(parents=True, exist_ok=True)
    correct_pos_path = experimeent_folder / f'correct_positives'
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
                            accuracy: float , pos_error: float, precision: float,
                            recall: float) -> None:
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
