import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import pyabf
from typing import Tuple

from minipeak import styles as ps

def _parse_args() -> Tuple[Path, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_folder', type=Path, help='path to experiment folder')
    parser.add_argument('--xls_file', type=Path, help='path to experiment xls file')
    parser.add_argument('--exp_name', type=str, help='name of experiment; abf file and '
                                                     'tab in excel file must match')
    args = parser.parse_args()
    return args.experiment_folder, args.xls_file, args.exp_name


def read_experiment(xls_file, exp_name) -> pd.DataFrame:
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


def read_abf(abf_file) -> pd.DataFrame:
    abf = pyabf.ABF(abf_file)
    time = abf.sweepX
    amplitude = abf.sweepY
    df = pd.DataFrame({'time':time, 'amplitude':amplitude})
    return df


def plot_experiment(xls_file, abf_file, exp_name) -> None:
    minis_df = read_experiment(xls_file, exp_name)
    timeserie_df = read_abf(abf_file)
    # plot minis_df and timeserie_df in smae graph with 'time' as x axis
    _, ax = plt.subplots()
    ax.plot(minis_df['time'], minis_df['amplitude'], 'o', label='minis')
    ax.plot(timeserie_df['time'], timeserie_df['amplitude'], label='timeserie')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('amplitude (mV)')
    ax.set_title(f'{exp_name} mini analysis')
    ax.legend()
    ps.show(block=True)


def main() -> None:
    ps.set_style('default')
    
    experiment_folder, xls_file, exp_name = _parse_args()
    xls_file = experiment_folder / xls_file
    abf_file = experiment_folder / f'{exp_name}.abf'
    plot_experiment(xls_file, abf_file, exp_name)


if __name__ == '__main__':
    """
    Main application when run as a script.
    """
    main()