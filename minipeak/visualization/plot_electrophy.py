import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pyabf

from minipeak import styles as ps


def _parse_args() -> argparse.Namespace:
    """ Parse the arguments provided through the command line. """
    parser = argparse.ArgumentParser()
    parser.add_argument('abf_file', type=Path, help='path to abf file')
    return parser.parse_args()


def _plot_electrophy(abf_file: Path) -> None:
    # load data from abf file
    abf = pyabf.ABF(abf_file)
    time = abf.sweepX
    amplitude = abf.sweepY
    # plot data
    _, ax = plt.subplots()
    ax.plot(time, amplitude, label='timeserie')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('amplitude (mV)')
    ax.set_title('mini analysis')
    ax.legend()
    ps.show(block=True)


def main() -> None:
    """ Main application when run from the command line interface. """
    ps.set_style('default')

    args = _parse_args()
    _plot_electrophy(args.abf_file)


if __name__ == '__main__':
    """ Main application when run as a script. """
    main()
