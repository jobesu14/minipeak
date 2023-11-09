import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from minipeak import styles as ps


def _parse_args() -> argparse.Namespace:
    """ Parse the arguments provided through the command line. """
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', type=Path, help='path to training_results.csv file')
    return parser.parse_args()


def _plot_training_evolution(csv_file: Path) -> None:
    training_df = pd.read_csv(csv_file)
    
    # split plot vertical into two sub plots
    ax1 = plt.subplot(211)
    ax1.plot(training_df["epoch"], training_df["accuracy"], label='accuracy')
    ax1.set_xlabel('epoch')
    ax1.set_title('training evolution')
    ax1.legend()
    
    ax2 = plt.subplot(212)
    ax2.plot(training_df["epoch"], training_df["loss"], label='loss')
    ax2.set_xlabel('epoch')
    ax2.legend()
    
    ps.show(block=True)


def main() -> None:
    """ Main application when run from the command line interface. """
    ps.set_style('default')

    args = _parse_args()
    _plot_training_evolution(args.csv_file)


if __name__ == '__main__':
    """ Main application when run as a script. """
    main()
