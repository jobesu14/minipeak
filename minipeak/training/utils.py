import json
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import  pandas as pd
import time
from typing import Tuple
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
