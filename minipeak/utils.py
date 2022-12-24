import  pandas   as   pd


def remove_low_freq_trend(serie: pd.Series, window: int) -> pd.DataFrame: 
     trend = serie.rolling(window = window, center = True).mean()
     return serie - trend


def expand_minis_to_electrophy(minis_df: pd.DataFrame, electrophy_df: pd.DataFrame) -> pd.DataFrame: 
    """ 
    Expand minis_df to match electrophy_df timestep. 
    """ 
    expended_minis_df = electrophy_df.copy()
    mini_id = 0
    t_next_mini = minis_df['time'].iloc[mini_id]
    for index, row in expended_minis_df.iterrows():
        if mini_id < len(minis_df) - 1 and t_next_mini <= row['time']:
            mini_id += 1
            t_next_mini = minis_df['time'].iloc[mini_id]
        else:
            expended_minis_df.at[index, 'amplitude'] = 0.0
    return expended_minis_df