import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from minipeak.preprocessing import load_training_data_from_csv
from minipeak import styles as ps


def _parse_args() -> argparse.Namespace:
    """ Parse the arguments provided through the command line. """
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', type=Path, help='path to experiment csv file')
    return parser.parse_args()


def _plot_training_data(experiment_df: pd.DataFrame, exp_name: str) \
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
    """ Main application when run from the command line interface. """
    ps.set_style('default')

    args = _parse_args()
    exp_name = args.csv_file.stem
    experiment_df = load_training_data_from_csv(args.csv_file)
    _plot_training_data(experiment_df, exp_name)


if __name__ == '__main__':
    """ Main application when run as a script. """
    main()
