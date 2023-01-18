import argparse
from pathlib import Path

from minipeak.custom_data_formating.mcabe_lab_utils import load_experiment_from_foder
from minipeak import styles as ps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('abf_folder', type=Path,
                        help='path to folder containing abf files')
    parser.add_argument('xls_file', type=Path, help='path to experiment xls file')
    parser.add_argument('csv_folder',
                        type=Path, help='path to folder containing csv files')
    parser.add_argument('--exp_name', type=str, help='name of experiment; abf file and '
                                                     'tab in excel file must match')
    parser.add_argument('--sampling_rate', type=int, default=100,
                        help='timeserie decimation ratio')
    parser.add_argument('--remove_trend_win_ms', type=int, default=100,
                        help='time window in ms to remove trend from timeserie')
    return parser.parse_args()


def _write_experiment_to_csv(xls_file: Path, abf_file: Path, csv_folder: Path, 
                             exp_name: str, sampling_rate:int,
                             remove_trend_win_ms: int) -> None:
    experiment_df = load_experiment_from_foder(xls_file, abf_file, exp_name,
                                               sampling_rate, remove_trend_win_ms)
    experiment_df.to_csv(csv_folder / f'{exp_name}.csv', index=False)


def main() -> None:
    ps.set_style('default')

    args = _parse_args()

    args.csv_folder.mkdir(parents=True, exist_ok=True)
    if exp_name is None:
        # find all abf files in experiment folder and write corresponding csv files
        abf_files = list(args.abf_folder.glob('*.abf'))
        for abf_file in abf_files:
            exp_name = abf_file.stem
            _write_experiment_to_csv(args.xls_file, abf_file, args.csv_folder, exp_name,
                                     args.sampling_rate, args.remove_trend_win_ms)
    else:
        abf_file = args.abf_folder / f'{exp_name}.abf'
        _write_experiment_to_csv(args.xls_file, abf_file, args.csv_folder, exp_name,
                                 args.sampling_rate, args.remove_trend,
                                 args.remove_trend_win_ms)


if __name__ == '__main__':
    """
    Main application when run as a script.
    """
    main()
