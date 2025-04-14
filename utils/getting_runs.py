# read runs off the csv from the rno-runtable
import pandas as pd
import os
import numpy as np

def get_df(file_path):
    """
    Reads the csv file and returns a dataframe
    Parameters:
    file_path (str): path to the csv file
    Returns:
    df (pd.DataFrame): dataframe containing the data from the csv file
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    # skip the first 3 rows and read the rest of the file
    df = pd.read_csv(file_path,  skiprows=3)
    return df

def get_runs(df, save_dir, pulser_type):
    """
    Runs rnogcopy to store combined.root files
    Parameters:
    df (pd.DataFrame): dataframe containing the data from the csv file
    save_dir (str): directory to save the combined.root files
    pulser_type (str): type of run
    """
    if pulser_type not in df['pulser'].unique():
        raise ValueError(f"Pulser type {pulser_type} not found in dataframe")
    for row in df[df['pulser'] == pulser_type].itertuples():
        st = row.station
        run = row.run
        print(f'getting run {run} for station {st}')
        dir_path = f'{save_dir}/station{st}/{pulser_type}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        os.chdir(dir_path)
        print(f'copying into {os.getcwd()}')
        os.system(f"rnogcopy run {st} {run} --server="chicago"")
    

# save_dir = '/data/user/sanyukta/rno_data/cal_pulser'
# df = get_df("quality_check__in-situ_pulsing_run_information.csv")
# get_runs(df, save_dir, 'fiber0')
# get_runs(df, save_dir, 'fiber1')