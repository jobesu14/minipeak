import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Tuple

from minipeak.preprocessing import load_experiment_from_csv
from minipeak import styles as ps


def _parse_args() -> Tuple[Path, Path, str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', type=Path, help='path to experiment csv file')
    args = parser.parse_args()
    return args.csv_file


def plot_experiment(experiment_df: pd.DataFrame, exp_name: str) \
        -> None:
    # plot minis_df and timeserie_df in same graph with 'time' as x axis
    _, ax = plt.subplots()
    ax.plot(experiment_df['time'], experiment_df['minis'], 'o', label='minis')
    ax.plot(experiment_df['time'], experiment_df['amplitude'], label='timeserie')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('amplitude (mV)')
    ax.set_title(f'{exp_name} mini analysis')
    ax.legend()
    ps.show(block=True)


def main() -> None:
    ps.set_style('default')
    
    csv_file = _parse_args()
    exp_name = csv_file.stem
    experiment_df = load_experiment_from_csv(csv_file)
    plot_experiment(experiment_df, exp_name)


if __name__ == '__main__':
    """
    Main application when run as a script.
    """
    main()
