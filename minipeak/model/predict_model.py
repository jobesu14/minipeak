import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import torch

from minipeak import styles as ps
from minipeak.network.cnn_model import CNN
from minipeak.preprocessing import split_into_overlapping_windows, read_abf, \
    remove_low_freq_trend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('abf_file', type=Path,
                        help='path to electrophy amplitude abf file')
    parser.add_argument('model_path', type=Path,
                        help='path to minipeak pytorch model file')
    parser.add_argument('--sampling_rate', type=int, default=100,
                        help='timeserie decimation ratio')
    parser.add_argument('--remove_trend_win_ms', type=int, default=100,
                        help='time window in ms to remove trend from timeserie')
    parser.add_argument('--window-size', type=int, default=80,
                        help='window size in ms (MUST match the model window size '
                        'used during training)')
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda')
    return parser.parse_args()


def _plot_experiment(experiment_df: pd.DataFrame, peaks_found: np.ndarray) \
        -> None:
    # plot minis_df and timeserie_df in same graph with 'time' as x axis
    _, ax = plt.subplots()
    ax.plot(experiment_df['amplitude'][peaks_found], 'o', label='mini peaks')
    ax.plot(experiment_df['amplitude'], label='amplitude')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('amplitude (mV)')
    ax.set_title(f'peaks detected')
    ax.legend()
    ps.show(block=True)


def _find_mini_peaks(model: CNN, amplitude: pd.DataFrame, window_size: int,
                     device: torch.device) -> np.ndarray:
    model.eval()

    windows = split_into_overlapping_windows(amplitude['amplitude'].values, window_size,
                                             int(window_size/2))
    peak_indices = []
    for win_num, window in enumerate(windows):
        # Predict the peak probability and peak time in the window with the CNN.
        torch_win = torch.from_numpy(window).reshape(-1, 1, window_size).float()
        torch_win = torch_win.to(device)
        pred = model(torch_win).cpu()

        # First value of NN output is the estimated peak probability.
        win_has_peak = bool(pred[0, 0] > 0.5)
        # Second value of NN output is the estimated peak position in percent of the window.
        win_peak_index = int(float(pred[0, 1]) * window_size)
        
        # If a peak has been predicted in this window, add the time of the peak computed 
        # from the begining of the timeserie.
        if win_has_peak:
            global_peak_index = int(win_num / 2.0 * window_size) + win_peak_index
            global_peak_index += amplitude.index[0] #  amplitude index doesn't start at 0
            peak_indices.append(global_peak_index)

    return np.array(peak_indices)


def main() -> None:
    ps.set_style('default')
    logging.basicConfig(level='INFO')
    args =_parse_args()

    device = \
        torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logging.info(f'Using device: {device}')

    # Load the data from the abf file and optionally perform some pre-processing.
    electrophy_df = read_abf(args.abf_file, sampling_rate=args.sampling_rate)
    electrophy_df['amplitude'] = \
        remove_low_freq_trend(electrophy_df['amplitude'],
                                window_ms=args.remove_trend_win_ms)
    electrophy_df.dropna(subset=['amplitude'], inplace=True)

    # Load the trained CNN model.
    model = CNN(window_size=args.window_size)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    # Find mini peaks time.
    peaks_time = _find_mini_peaks(model, electrophy_df, args.window_size, device)

    # Plot the result of the mini peaks detection
    _plot_experiment(electrophy_df, peaks_time)
    logging.info(f'Found {len(peaks_time)} mini peaks in {args.abf_file.name}')


if __name__ == '__main__':
    """
    Main application when run as a script.
    """
    main()
