from pathlib import Path
import pandas as pd

from minipeak.preprocessing import read_abf, remove_low_freq_trend


def _group_minis_and_electrophy(minis_df: pd.DataFrame, electrophy_df: pd.DataFrame) \
        -> pd.DataFrame:
    """
    Combine the mini-peaks amplitude dataframe and the electrophysiology dataframe into a
    single dataframe where each row is a time point with the electrophysiology amplitude
    and the corresponding mini-peaks amplitude. Mini-peaks amplitude is set to 0.0 if no
    peak for that time point.

    :param minis_df: dataframe with mini-peaks amplitude and time
    :param electrophy_df: dataframe with electrophysiology amplitude and time
    :return: dataframe with electrophysiology amplitude and mini-peaks amplitude
    """
    experiment_df = electrophy_df.copy()
    # remove row with no 'amplitude' data
    experiment_df.dropna(subset=['amplitude'], inplace=True)
    experiment_df['minis'] = 0.0  # add minis column filled with zeros

    mini_id = 0
    t_next_mini = minis_df['time'].iloc[mini_id]
    for index, row in experiment_df.iterrows():
        if mini_id < len(minis_df) - 1 and t_next_mini <= row['time']:
            mini_id += 1
            t_next_mini = minis_df['time'].iloc[mini_id]
            experiment_df.at[index, 'minis'] = row['amplitude']
    return experiment_df


def _read_minis(xls_file: Path, exp_name: str) -> pd.DataFrame:
    """
    Read the mini-peaks amplitude from the excel file. The excel file must have a sheet
    named as the experiment name. The sheet must have the following columns:
    - time: time of the mini-peak in ms in column B
    - amplitude: amplitude of the mini-peak in mV in column C
    - baseline: baseline of the mini-peak in mV in column G

    :param xls_file (Path): path to the excel file for this experiment
    :param exp_name (str): name of the experiment (must correspnd to one of the sheet
    name in the excel file)
    :return: dataframe with mini-peaks amplitude and time
    """
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


def load_experiment_from_foder(xls_file: Path, abf_file: Path, exp_name: str,
                               sampling_rate: int, remove_trend_win_ms: int) \
        -> pd.DataFrame:
    """
    Load the experiment data from the excel file and the electrophysiology data from the
    abf file. The excel file must have a sheet named as the experiment name. The sheet
    must have the following columns:
    - time: time of the mini-peak in ms in column B
    - amplitude: amplitude of the mini-peak in mV in column C
    - baseline: baseline of the mini-peak in mV in column G

    :param xls_file (Path): path to the excel file for this experiment
    :param abf_file (Path): path to the abf file for this experiment
    :param exp_name (str): name of the experiment (must correspnd to one of the sheet
    name in the excel file)
    :param sampling_rate (int): dwosamplig ratio for the electrophysiology data
    :param remove_trend_win_ms (int): size of the window in ms to remove the trend from
    the electrophysiology signal
    """
    minis_df = _read_minis(xls_file, exp_name)
    electrophy_df = read_abf(abf_file, sampling_rate=sampling_rate)
    electrophy_df['amplitude'] = remove_low_freq_trend(electrophy_df['amplitude'],
                                                       window_ms=remove_trend_win_ms)
    experiment_df = _group_minis_and_electrophy(minis_df, electrophy_df)
    return experiment_df
