import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Tuple

from minipeak import styles as ps
from minipeak.preprocessing import read_abf, remove_low_freq_trend


def _parse_args() -> Tuple[Path, int, bool, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument('abf_file', type=Path,
                        help='path to electrophy amplitude abf file')
    parser.add_argument('--sampling_rate', type=int, default=100,
                        help='timeserie decimation ratio')
    parser.add_argument('--remove_trend', action='store_true', default=True,
                        help='remove low frequency trend from timeserie')
    parser.add_argument('--remove_trend_win_ms', type=int, default=100,
                        help='time window in ms to remove trend from timeserie')
    args = parser.parse_args()
    return args.abf_file, args.sampling_rate, args.remove_trend, args.remove_trend_win_ms


def plot_experiment(experiment_df: pd.DataFrame, exp_name: str) \
        -> None:
    # plot minis_df and timeserie_df in same graph with 'time' as x axis
    _, ax = plt.subplots()
    # ax.plot(experiment_df['time'], experiment_df['minis'], 'o', label='minis')
    ax.plot(experiment_df['time'], experiment_df['amplitude'], label='timeserie')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('amplitude (mV)')
    ax.set_title(f'{exp_name} peaks detection')
    ax.legend()
    ps.show(block=True)


def main() -> None:
    ps.set_style('default')
    
    abf_file, sampling_rate, remove_trend, remove_trend_win_ms = _parse_args()
    electrophy_df = read_abf(abf_file, sampling_rate=sampling_rate)
    if remove_trend:
        electrophy_df['amplitude'] = remove_low_freq_trend(electrophy_df['amplitude'],
                                                           window_ms=remove_trend_win_ms)
    plot_experiment(electrophy_df, "test")


if __name__ == '__main__':
    """
    Main application when run as a script.
    """
    main()
