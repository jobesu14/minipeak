import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Tuple

from minipeak.utils import load_experiment_from_foder
from minipeak import styles as ps


def _parse_args() -> Tuple[Path, Path, str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_folder', type=Path, help='path to experiment folder')
    parser.add_argument('--xls_file', type=Path, help='path to experiment xls file')
    parser.add_argument('--exp_name', type=str, help='name of experiment; abf file and '
                                                     'tab in excel file must match')
    parser.add_argument('--remove_trend', action='store_true', default=False,
                        help='apply low pass filter to timeserie')
    args = parser.parse_args()
    return args.experiment_folder, args.xls_file, args.exp_name, args.remove_trend


def main() -> None:
    ps.set_style('default')
    
    experiment_folder, xls_file, exp_name, remove_trend = _parse_args()
    xls_file = experiment_folder / xls_file
    abf_file = experiment_folder / f'{exp_name}.abf'
    experiment_df = load_experiment_from_foder(xls_file, abf_file, exp_name, remove_trend)
    experiment_df.to_csv(experiment_folder / f'{exp_name}.csv', index=False)


if __name__ == '__main__':
    """
    Main application when run as a script.
    """
    main()
