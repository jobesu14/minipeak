import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pyabf

from minipeak import styles as ps


def _parse_args() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument('abf_file', type=Path, help='path to abf file')
    args = parser.parse_args()
    return args.abf_file


def plot_electrophy(abf_file) -> None:
    # load data from abf file
    abf = pyabf.ABF(abf_file)
    time = abf.sweepX
    amplitude = abf.sweepY
    # plot data
    _, ax = plt.subplots()
    ax.plot(time, amplitude, label='timeserie')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('amplitude (mV)')
    ax.set_title(f'mini analysis')
    ax.legend()
    ps.show(block=True)


def main() -> None:
    ps.set_style('default')

    abf_file = _parse_args()
    plot_electrophy(abf_file)


if __name__ == '__main__':
    """
    Main application when run as a script.
    """
    main()
