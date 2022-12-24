import numbers
import os
from typing import Callable, List, Optional, Union
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplcursors
from mplcursors import Selection


def show(block: bool = False,
         cursor_annotation: Optional[Callable[[Selection], None]] = None) -> None:
    """
    Improvement over the Matplotlib :func:`~matplotlib.pyplot.show` function, which does
    the following:

    * Fixes minor grid of Matplotlib (if activated)
    * Creates point picking, disabled by default (activate it with keybord key ``e``)
    * Calls the Matplotlib :func:`~matplotlib.pyplot.show` function

    :param block: Blocks the execution of further code. Should be set to true, if there is
        no further code, in order to prevent the program from shutting down.
    """
    # Fix grid of all axis in all figures
    fignums = plt.get_fignums()
    fig_list = [plt.figure(fignum) for fignum in fignums]
    for fig in fig_list:
        ax_list = fig.axes
        for ax in ax_list:
            if ax.name != '3d':
                _fix_grid(ax)
    # Create point picking
    cursor = mplcursors.cursor(multiple=True)
    if cursor_annotation is not None:
        cursor.connect('add', cursor_annotation)
    cursor.enabled = False
    # Show plots
    plt.show(block=block)
    plt.pause(.001)


def set_style(style_name: Union[str, List[str]], ignore_if_fails: bool = True) -> None:
    """
    Set a specific style (should be called just after the import of this module). Styles
    can be stacked on each others too!

    See available ones with: :func:`matplotlib.pyplot.style.available`
    Info here:

    * https://matplotlib.org/tutorials/introductory/customizing.html
    * https://matplotlib.org/examples/showcase/anatomy.html

    Examples::

        >>> plot_style.set_style('default')  # Default style
        >>> plot_style.set_style('default', 'publish'])  # Publish style

    :param style_name: Name of the style(s) (file name in ``styles/`` folder, without the
        file extension)
    :param ignore_if_fails: Do not throw an OSError, if the function fails. Needed when
        using in combination with pyinstaller.
    """
    try:
        if isinstance(style_name, str):
            style_name = [style_name]
        styles_list = [os.path.join(os.path.dirname(__file__), 'styles/' + style
                                    + '.mplstyle') for style in style_name]
        plt.style.use(styles_list)
    except OSError as error:
        if ignore_if_fails:
            print(error, file=sys.stderr)
        else:
            raise error


def set_publish_style() -> None:
    """
    Convenience function to apply the publish style more easily, since it depends on two
    styles.
    """
    set_style(['default', 'publish'])


def _fix_grid(ax: mpl.axes.Axes) -> None:
    """Fix minor grid style, not appearing when 'both' grids are set."""
    # Check if minor grid is activated
    # (Ugly: https://stackoverflow.com/questions/23246021/matplotlib-check-if-grid-is-on)
    if (hasattr(ax.xaxis, '_gridOnMinor') and ax.xaxis._gridOnMinor) \
            or \
            (hasattr(ax.xaxis, '_minor_tick_kw') and ax.xaxis._minor_tick_kw['gridOn']):
        # Minor ticks must be enabled to show minor grid, but we don't want them!
        # plt.minorticks_on()
        ax.minorticks_on()
        ax.tick_params(which='minor', length=0)
        # Not possible to set minor grid style in style sheet...
        # (See here: https://github.com/matplotlib/matplotlib/issues/13919)
        ax.grid(b=True, which='minor', color='0.1', alpha=0.25, linestyle=':')


def scale_fonts(scale_factor: float) -> None:
    '''Scale default plot font sizes according to the scale factor.'''
    if scale_factor <= 0:
        raise ValueError(
            f"figure font scale factor has to be positive, got: {scale_factor}")

    p = mpl.rcParams

    if isinstance(p['figure.titlesize'], numbers.Number):
        plt.rc('figure', titlesize=p['figure.titlesize'] * scale_factor)  # type: ignore

    plt.rc('font', size=p['font.size'] * scale_factor)
    plt.rc('axes', titlesize=p['axes.titlesize'] * scale_factor)
    plt.rc('axes', labelsize=p['axes.labelsize'] * scale_factor)
    plt.rc('xtick', labelsize=p['xtick.labelsize'] * scale_factor)
    plt.rc('ytick', labelsize=p['ytick.labelsize'] * scale_factor)
    plt.rc('legend', fontsize=p['legend.fontsize'] * scale_factor)


def default_colors() -> List[str]:
    """
    Get the default colors associated with the current style.

    :return: List of all the default colors.
    """
    return [color['color'] for color in list(mpl.rcParams['axes.prop_cycle'])]
