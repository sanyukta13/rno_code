# read runs off the csv from the rno-runtable
import pandas as pd
import os
import numpy as np
import re

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
        os.system(f"rnogcopy run {st} {run}")

def add_attns(df, source_dir):
    """
    Adds attenuation values to the dataframe
    Parameters:
    df (pd.DataFrame): dataframe containing the data from the csv file
    source_dir (str): directory containing the run files
    Returns:
    df (pd.DataFrame): dataframe with attenuation values added
    """
    df['attenuation'] = 0
    for row in df.itertuples():
        st = row.station
        run = row.run
        text_file = f'{source_dir}/station{st}/run{run}/aux/comment.txt'
        if not os.path.exists(text_file):
            print(f"{text_file} does not exist, setting attenuation to 0")
        else:
            try:
                with open(text_file, 'r') as file:
                    content = file.read()
                    # Use regex to extract the attenuation value
                    match = re.search(r'(\d+)dB attenuation', content)
                    if match:
                        df.at[row.Index, 'attenuation'] = int(match.group(1))
                    else:
                        print(f"No attenuation value found in {text_file}, setting to 0")
            except Exception as e:
                print(f"Error reading {text_file}: {e}")
    return df

# save_dir = '/data/user/sanyukta/rno_data/cal_pulser'
source_dir = '/data/desy-mirror/inbox'
df = get_df("quality_check__in-situ_pulsing_run_information.csv")
add_attns(df, source_dir).to_csv("att_info.csv", index=False)
# get_runs(df, save_dir, 'fiber0')
# get_runs(df, save_dir, 'fiber1')