import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Tuple

from minipeak.utils import load_experiment_from_foder
from minipeak import styles as ps


def _parse_args() -> Tuple[Path, Path, str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument('abf_folder', type=Path,
                        help='path to folder containing abf files')
    parser.add_argument('csv_folder',
                        type=Path, help='path to folder containing csv files')
    parser.add_argument('xls_file', type=Path, help='path to experiment xls file')
    parser.add_argument('--exp_name', type=str, help='name of experiment; abf file and '
                                                     'tab in excel file must match')
    parser.add_argument('--remove_trend', action='store_true', default=False,
                        help='apply low pass filter to timeserie')
    args = parser.parse_args()
    return args.abf_folder, args.csv_folder, args.xls_file, args.exp_name, args.remove_trend

def _write_experiment_to_csv(xls_file: Path, abf_file: Path, csv_folder: Path, 
                             exp_name: str, remove_trend: bool) -> None:
    experiment_df = load_experiment_from_foder(xls_file, abf_file, exp_name, remove_trend)
    experiment_df.to_csv(csv_folder / f'{exp_name}.csv', index=False)

def main() -> None:
    ps.set_style('default')
    
    abf_folder, csv_folder, xls_file, exp_name, remove_trend = _parse_args()
    if exp_name is None:
        # find all abf files in experiment folder and write corresponding csv files
        abf_files = list(abf_folder.glob('*.abf'))
        for abf_file in abf_files:
            exp_name = abf_file.stem
            _write_experiment_to_csv(xls_file, abf_file, csv_folder, exp_name, remove_trend)
    else:
        abf_file = abf_folder / f'{exp_name}.abf'
        _write_experiment_to_csv(xls_file, abf_file, csv_folder, exp_name, remove_trend)

if __name__ == '__main__':
    """
    Main application when run as a script.
    """
    main()
