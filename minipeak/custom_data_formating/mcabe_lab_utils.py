from pathlib import Path
import pandas as pd

from minipeak.preprocessing import read_abf, remove_low_freq_trend


def _group_minis_and_electrophy(minis_df: pd.DataFrame, electrophy_df: pd.DataFrame) \
        -> pd.DataFrame: 
    """ 
    Expand minis_df to match electrophy_df timestep. 
    """
    experiment_df = electrophy_df.copy()
    # remove row with no 'amplitude' data
    experiment_df.dropna(subset=['amplitude'], inplace=True)
    experiment_df['minis'] = 0.0 ## add minis column filled with zeros

    mini_id = 0
    t_next_mini = minis_df['time'].iloc[mini_id]
    for index, row in experiment_df.iterrows():
        if mini_id < len(minis_df) - 1 and t_next_mini <= row['time']:
            mini_id += 1
            t_next_mini = minis_df['time'].iloc[mini_id]
            experiment_df.at[index, 'minis'] = row['amplitude']
    return experiment_df



def _read_minis(xls_file: Path, exp_name: str) -> pd.DataFrame:
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
                               sampling_rate: int, remove_trend: bool,
                               remove_trend_win_ms: int) -> pd.DataFrame:
    minis_df = _read_minis(xls_file, exp_name)
    electrophy_df = read_abf(abf_file, sampling_rate=sampling_rate)
    if remove_trend:
        electrophy_df['amplitude'] = remove_low_freq_trend(electrophy_df['amplitude'],
                                                           window_ms=remove_trend_win_ms)
    experiment_df = _group_minis_and_electrophy(minis_df, electrophy_df)
    return experiment_df
