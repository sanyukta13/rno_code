import numpy as np
import os
from os import listdir
from os.path import isfile, join
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import NuRadioReco
from NuRadioReco.framework.base_trace import BaseTrace
from NuRadioReco.utilities import units, fft
from NuRadioReco.utilities.fft import time2freq, freq2time
from NuRadioReco.modules import channelAddCableDelay, channelBandPassFilter, sphericalWaveFitter
from NuRadioReco.detector import detector
from NuRadioReco.modules.io.RNO_G import readRNOGDataMattak
import astropy.time
import logging
import json
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
from datetime import datetime
from scipy.signal import correlate

# Suppress the AstropyDeprecationWarning
warnings.filterwarnings('ignore', category=AstropyDeprecationWarning)
AddCableDelay = channelAddCableDelay.channelAddCableDelay()
BandPassFilter = channelBandPassFilter.channelBandPassFilter()
DET = detector.Detector(json_filename = "/data/user/sanyukta/rno_data/data/json/RNO_season_2023.json")
DET.update(datetime.now())
DATA_PATH_ROOT = '/data/user/sanyukta/rno_data/cal_pulser'

# reader for read_raw_traces
def read_root_file(station_number, run, selectors=[], sampling_rate=3.2):
    '''
    Reads the root file of the specified station and run and returns a NuRadioReco.modules.io.RNO_G.readRNOGData reader object.
    1) Create a NuRadioReco.modules.io.RNO_G.readRNOGData object
    2) Find out fiber number of this station number and run
    3) Generate filepath of the run using station number, fiber number and run
    4) Begin reading this filepath using the readRNOGData object
    
    Parameters
    ----------
    station_id : int
        station id
    run : int
        run number
    selectors : list, optional
        list of selectors or filters you want to apply on the events read, default is an empty list
    sampling_rate : float, optional
        sampling rate in GHz at which to read volts, default is 3.2

    Returns
    -------
    reader : NuRadioReco.modules.io.RNO_G.readRNOGData
        readRNOGData object that can be used to fetch volts and times lists to construct a dataset for the run, None if file not found
    '''
    
    reader = readRNOGDataMattak.readRNOGData()
    fiber_number = get_fiber_for_run(station_id=station_number, run=run)
    print(fiber_number)
    if fiber_number is not None:
        file = DATA_PATH_ROOT + '/st' + str(station_number) + '/fiber' + str(fiber_number) + '/st' + str(station_number) + '_run' + str(run) + '.root'
    else:
        print("File not found")
        return None
    print(file)
    reader.begin(dirs_files=file, selectors=selectors, overwrite_sampling_rate=sampling_rate)
    return reader

# pulsed events only (cal pulser runs)
def read_raw_traces(station_id, run, sampling_rate=3.2, pulse_rms_factor=6):
    '''
    Returns times, volts, events in the following format
    volts - [
-> event 1  {
    -> channel  0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
    -> channel  1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ...
    -> channel 7: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
          -> time  t0 t1 t2 t3 t4 t5 t6 t7 t8 t9
            },
...            
-> event n  {
    -> channel  0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
    -> channel  1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ...
    -> channel 20: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
          -> time  t0 t1 t2 t3 t4 t5 t6 t7 t8 t9            
            },        
    ]
    '''
    selectors = [lambda event_info: (event_info.sysclk - event_info.sysclkLastPPS[0]) % (2**32) <= CAL_PULSER_THRESHOLD]
    reader = read_root_file(station_number=station_id, run=run, selectors=selectors)
    event_ids, times, volts = [], [], []

    # Read all events after cal pulser threshold is applied
    
    for index, event in enumerate(reader.run()):
        times.append({})
        volts.append({})
        event_ids.append(event.get_id())
        station = event.get_station(station_id=station_id)
        AddCableDelay.run(event, station, DET, mode='subtract')
        for ch in PA_CHS:
            channel = station.get_channel(ch)
            times[index][ch] = channel.get_times()
            volts[index][ch] = channel.get_trace()

    # Filter out all events that don't have a pulse in the receiver channel closest to the cal pulser
    
    reference_channel = get_ref_ch(station_id=station_id, run=run)
    no_pulse_event_index = []
    for index in range(len(volts)):
        waveform_ref_i = volts[index][reference_channel]
        waveform_mean_ref_i = waveform_ref_i - np.mean(waveform_ref_i) #dc offset
        pulse_found = False
        for i in range(2, len(waveform_mean_ref_i)):
            if np.abs(waveform_mean_ref_i[i]) > np.mean(waveform_mean_ref_i[0:i]) + pulse_rms_factor*np.std(waveform_mean_ref_i[0:i]):
                pulse_found = True
                break
        if not pulse_found:
            no_pulse_event_index.append(index)
    no_pulse_event_index.sort(reverse=True)
    for index in no_pulse_event_index:
        volts.pop(index)
        times.pop(index)
        event_ids.pop(index)
    return event_ids, times, volts

# basic reader for rno-g root files
def basic_read_root(path_to_root, selectors = [], sampling_rate = 3.2):
    '''
    Reads the root file and returns a NuRadioReco.modules.io.RNO_G.readRNOGData reader object.
    1) Create a NuRadioReco.modules.io.RNO_G.readRNOGData object
    2) Begin reading filepath using the readRNOGData object
    
    Parameters
    ----------
    path_to_root : str
        path to the root file
    selector : list, optional
        list of selectors or filters you want to apply on the events read, default is an empty list
    sampling_rate : float, optional
        sampling rate in GHz at which to read volts, default is 3.2

    Returns
    -------
    reader : NuRadioReco.modules.io.RNO_G.readRNOGData
        readRNOGData object that can be used to fetch volts and times lists to construct a dataset for the run, None if file not found
    '''
    # initialize reader
    reader = readRNOGDataMattak.readRNOGData()
    if not (os.path.isfile(path_to_root)):
        print("File not found")
        return None
    print(f"\n reading {path_to_root} ......")
    reader.begin(dirs_files = path_to_root, selectors = selectors, overwrite_sampling_rate = sampling_rate)
    return reader

def get_eventsvoltstraces(reader, band_pass = 0):
    '''
    Parameters
    ----------
    reader : NuRadioReco.modules.io.RNO_G.readRNOGData
        readRNOGData object that can be used to fetch volts and times lists to construct a dataset for the run
    band_pass : int, optional
        0 if band pass filter is not to be applied, 1 if band pass filter is to be applied, default is 0
    Returns times, volts, events in the following format
    volts - [
-> event 1  {
    -> channel  0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
    -> channel  1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ...
    -> channel 7: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
          -> time  t0 t1 t2 t3 t4 t5 t6 t7 t8 t9
            },
...            
-> event n  {
    -> channel  0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
    -> channel  1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ...
    -> channel 20: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
          -> time  t0 t1 t2 t3 t4 t5 t6 t7 t8 t9            
            },        
    ]
    '''
    event_ids, times, volts = [], [], []    
    for index, event in enumerate(reader.run()):
        times.append({})
        volts.append({})
        event_ids.append(event.get_id())
        station_id = event.get_station_ids()[0]
        station = event.get_station(station_id)
        AddCableDelay.run(event, station, DET, mode='subtract')
        if band_pass:
            BandPassFilter.run(event, station, DET, passband = [175*units.MHz, 750*units.MHz])
        for channel in station.iter_channels():
            times[index][channel.get_id()] = channel.get_times()
            volts[index][channel.get_id()] = channel.get_trace()
    return event_ids, times, volts


def main():
    # Example usage of read_raw_traces
    events, times, volts = get_eventsvoltstraces(basic_read_root('/data/user/sanyukta/rno_data/cal_pulser/st11/fiber0/st11_run1726.root'), band_pass=1)
    print(volts[0][0]) 
if __name__ == "__main__":
    main()