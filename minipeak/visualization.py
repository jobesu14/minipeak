import matplotlib.pyplot as plt
import pandas as pd

from minipeak import styles as ps


def plot_training_curves(training_results_df: pd.DataFrame, plot_loss: bool = False) \
        -> None:
    ps.set_style('default')
    # plot training results
    _, ax = plt.subplots()
    ax.set_title(f'training results')
    ax.set_xlabel('epoch')
    ax.plot(training_results_df['epoch'], training_results_df['accuracy'],
            label='accuracy')
    if plot_loss:
        ax.plot(training_results_df['epoch'], training_results_df['loss'], label='loss')
        ax.set_ylabel('loss/accuracy')
    else:
        ax.set_ylabel('accuracy')
    ax.legend()
    ps.show(block=True)