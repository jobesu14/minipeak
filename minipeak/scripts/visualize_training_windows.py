import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

from minipeak import styles as ps
from minipeak.utils import load_windowed_dataset, convert_minis_to_amplitude


def _parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description='Plot the training data windows for a given experiment.')
    parser.add_argument('csv_file', type=Path,
                        help='path to the preprocessed minis csv file.')
    parser.add_argument('--window-size', type=int, default=100,
                        help='window size in ms')
    return parser.parse_args()


def plot_training_windows(amp_win: np.ndarray, minis_win: np.ndarray) -> None:
    # load data from csv file
    min_amplitude = min([min(amp_win) for amp_win in amp_win])
    max_amplitude = max([max(amp_win) for amp_win in amp_win])
    
    # iterate over all windows andn plot them
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


def main() -> None:
    ps.set_style('default')

    args = _parse_args()
    amp_win, minis_win = load_windowed_dataset(args.csv_file,
                                               args.window_size,
                                               int(args.window_size/2))
    plot_training_windows(amp_win, minis_win)


if __name__ == '__main__':
    """
    Main application when run as a script.
    """
    main()
