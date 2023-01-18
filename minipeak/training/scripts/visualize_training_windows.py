import argparse
from pathlib import Path

from minipeak import styles as ps
from minipeak.preprocessing import load_windowed_dataset
from minipeak.training.visualization import plot_training_windows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot the training data windows for a given experiment.')
    parser.add_argument('csv_file', type=Path,
                        help='path to the preprocessed minis csv file.')
    parser.add_argument('--window-size', type=int, default=100,
                        help='window size in ms')
    return parser.parse_args()


def main() -> None:
    ps.set_style('default')

    args = _parse_args()
    amp_win, minis_win = load_windowed_dataset(args.csv_file,
                                               args.window_size)
    plot_training_windows(amp_win, minis_win)


if __name__ == '__main__':
    """
    Main application when run as a script.
    """
    main()
