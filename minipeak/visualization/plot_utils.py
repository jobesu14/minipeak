import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from minipeak import styles as ps
from minipeak.utils import convert_minis_to_amplitude


def plot_training_windows(amp_win: np.ndarray, minis_win: np.ndarray) -> None:
    # load data from csv file
    min_amplitude = min([min(amp_win) for amp_win in amp_win])
    max_amplitude = max([max(amp_win) for amp_win in amp_win])
    
    # iterate over all windows and plot them
    for amplitude, minis in zip(amp_win, minis_win):
        peaks_time, peaks_amp = convert_minis_to_amplitude(amplitude, minis)
        # plot data
        _, ax = plt.subplots()
        ax.plot(amplitude, label='amplitude')
        ax.plot(peaks_time, peaks_amp, 'o', label='minis')
        # set plot y axis range
        ax.set_ylim(min_amplitude, max_amplitude)
        ax.set_ylabel('amplitude (mV)')
        ax.set_title(f'training windows')
        ax.legend()
        ps.show(block=True)


def plot_training_curves(training_results_df: pd.DataFrame, plot_loss: bool = False) \
        -> None:
    ps.set_style('default')
    # plot training results
    _, ax = plt.subplots()
    ax.set_title(f'training results')
    ax.set_xlabel('epoch')
    ax.plot(training_results_df['epoch'], training_results_df['accuracy'],
            label='accuracy')
    if plot_loss:
        ax.plot(training_results_df['epoch'], training_results_df['loss'], label='loss')
        ax.set_ylabel('loss/accuracy')
    else:
        ax.set_ylabel('accuracy')
    ax.legend()
    ps.show(block=True)