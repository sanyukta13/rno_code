import numpy as np
import os, sys
from os import listdir
from os.path import isfile, join
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import NuRadioReco
from NuRadioReco.framework.base_trace import BaseTrace
from NuRadioReco.utilities import units, fft, signal_processing
from NuRadioReco.utilities.fft import time2freq, freq2time
from NuRadioReco.modules import channelAddCableDelay, channelBandPassFilter, sphericalWaveFitter
from NuRadioReco.detector import detector
from NuRadioReco.modules.io.RNO_G import readRNOGDataMattak
import astropy.time, logging, json, warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
from datetime import datetime
from scipy.signal import correlate

sys.path.append(os.path.abspath('/home/sanyukta/software/source/analysis-tools/rnog_analysis_tools'))
from glitch_unscrambler.glitch_detection_per_event import diff_sq, is_channel_scrambled
from glitch_unscrambler.glitch_unscrambler import unscramble

# Suppress the AstropyDeprecationWarning
warnings.filterwarnings('ignore', category=AstropyDeprecationWarning)

AddCableDelay = channelAddCableDelay.channelAddCableDelay()
BandPassFilter = channelBandPassFilter.channelBandPassFilter()
DET = detector.Detector(json_filename = "/home/sanyukta/software/source/NuRadioMC/NuRadioReco/detector/RNO_G/RNO_season_2024.json")
DET.update(datetime.now())
DATA_PATH_ROOT = '/data/user/sanyukta/rno_data/cal_pulser'

# basic reader for rno-g root files
def basic_read_root(path_to_root, selectors = [], sampling_rate = 3.2, mattak_kwargs = {}):
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
    mattak_kwargs : dict, optional
        dictionary of keyword arguments to be passed to the mattak reader, default is an empty dict

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
    reader.begin(dirs_files = path_to_root, selectors = selectors, overwrite_sampling_rate = sampling_rate, mattak_kwargs = mattak_kwargs)
    return reader

def get_ch_pos(station_id, ch):
    '''
    Gets the position of a channel in a station
    
    Parameters
    ----------
    station_id : int
        station id
    ch : int
        channel

    Returns
    -------
    x_rx : float
    y_rx : float
    z_rx : float
    '''
    receiver_channel = DET.get_channel(station_id, ch)
    x_rx = receiver_channel['ant_position_x']
    y_rx = receiver_channel['ant_position_y']
    z_rx = receiver_channel['ant_position_z']
    return x_rx, y_rx, z_rx

def get_fiber_for_run(station_id, run):
    '''
    Gets the fiber number for a station and a run
    
    Parameters
    ----------
    station_id : int
        station id
    run : int
        run number

    Returns
    -------
    fiber_number : int
        fiber number if run is found in <DATA_PATH_ROOT>/st<station_id>/fiber<fiber_number>/st<station_id>_run<run>.root
        else None
    '''
    fiber_number = None
    atts = [0, 5, 10, 20]
    for att in atts:
        file_0 = DATA_PATH_ROOT + '/station' + str(station_id) + '/fiber0/' + str(att) + 'dB/station' + str(station_id) + '_run' + str(run) + '_combined.root'
        file_1 = DATA_PATH_ROOT + '/station' + str(station_id) + '/fiber1/' + str(att) + 'dB/station' + str(station_id) + '_run' + str(run) + '_combined.root'
        if os.path.isfile(file_0):
            fiber_number = 0
            break
        elif os.path.isfile(file_1):
            fiber_number = 1
            break
    
    if fiber_number is None:
        print(f"File not found for station {station_id} and run {run}")
        return None
    
    return fiber_number

def get_cp_pos(station_id, run):
    '''
    Gets the position of the calibration pulser for a station and a run
    
    Parameters
    ----------
    station_id : int
        station id
    run : int
        run number

    Returns
    -------
    x_rx : float
    y_rx : float
    z_rx : float
    '''
    
    cal_pulser = DET.get_device(station_id, get_fiber_for_run(station_id, run))
    x_tx = cal_pulser['ant_position_x']
    y_tx = cal_pulser['ant_position_y']
    z_tx = cal_pulser['ant_position_z']
    return x_tx, y_tx, z_tx

def get_ref_ch(station_id, run):
    '''
    Gets the channel id of the receiver channel that is closest to the calibration pulser in depth for a station and a run
    
    Parameters
    ----------
    station_id : int
        station id
    run : int
        run number

    Returns
    -------
    ref_ch : int
        channel id of the receiver channel that is closest to the calibration pulser
    '''
    PA_CHS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    _, _, z_tx = get_cp_pos(station_id, run=run)
    channel_depths = []
    for ch in PA_CHS:
        _, _, z_rx = get_ch_pos(station_id=station_id, ch=ch)
        channel_depths.append(z_rx)
    channel_depths = np.array(channel_depths)
    ref_ch = PA_CHS[np.abs(channel_depths - z_tx).argmin()]
    return ref_ch

def get_eventsvoltstraces(reader, band_pass = 0, pulse_filter = 0, pulse_rms_factor = 6,
                          freq_band_filter=None, cable_delay=1, glitch_filter=1):
    '''
    Parameters
    ----------
    reader : NuRadioReco.modules.io.RNO_G.readRNOGData
        readRNOGData object that can be used to fetch volts and times lists to construct a dataset for the run
    band_pass : int, optional
        0 if band pass filter is not to be applied, 1 if band pass filter is to be applied, default is 0
    pulse_filter : int, optional
        0 if pulse filter is not to be applied, 1 if pulse filter is to be applied, default is 0
    pulse_rms_factor : int, optional
        pulse rms factor to be used for pulse filter, default is 6
    freq_band_filter : tuple, optional
        frequency band filter to be applied, default is None
    cable_delay : int, optional
        cable delay to be applied, default is 1
    glitch_filter : int, optional
        0 if glitch filter is not to be applied, 1 if glitch filter is to be applied, default is 1
    ----------
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

        glitch_chs = []
        if glitch_filter:
            for channel in station.iter_channels():
                # glitch detection
                if is_channel_scrambled(channel.get_trace())>0:
                    glitch_chs.append(channel.get_id())
                
        if cable_delay:
            AddCableDelay.run(event, station, DET, mode='subtract')

        if band_pass:
            BandPassFilter.run(event, station, DET, passband = [175*units.MHz, 750*units.MHz])
        
        if freq_band_filter is not None:
            BandPassFilter.run(event, station, DET, passband = [freq_band_filter[0]*units.MHz, freq_band_filter[1]*units.MHz])

        pulse_found = True
        if pulse_filter:
            run_id = event.get_run_number()
            reference_channel = get_ref_ch(station_id, run_id)
            for channel in station.iter_channels():
                if channel.get_id() == reference_channel:
                    waveform_ref_i = channel.get_trace()
            waveform_mean_ref_i = waveform_ref_i - np.mean(waveform_ref_i)
            pulse_found = False
            if np.max(np.abs(waveform_mean_ref_i)) > pulse_rms_factor*np.std(waveform_mean_ref_i):
                if np.argmax(np.abs(waveform_mean_ref_i))>150 and np.argmax(np.abs(waveform_mean_ref_i))<1898:
                    pulse_found = True
        
        if pulse_found:
            for channel in station.iter_channels():
                if channel.get_id() in glitch_chs:
                    #disregard the whole event if any channel is scrambled
                    times[index] = None
                    volts[index] = None
                    event_ids.pop(-1)
                    break
                else:
                    volts[index][channel.get_id()] = channel.get_trace()
                    times[index][channel.get_id()] = channel.get_times()

        else:
            times[index] = None
            volts[index] = None
            event_ids.pop(-1)

    filtered_data = [(volt, time) for volt, time in zip(volts, times) if volt is not None]
    if filtered_data:
        v, t = zip(*filtered_data)
        v, t = list(v), list(t)  # Convert back to lists if needed
    else:
        v, t = [], []  # Assign empty lists if no valid data is found
    return event_ids, t, v