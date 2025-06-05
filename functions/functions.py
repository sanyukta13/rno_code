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
from scipy.signal import correlate, hilbert
from scipy.integrate import simpson

# Suppress the AstropyDeprecationWarning
warnings.filterwarnings('ignore', category=AstropyDeprecationWarning)
AddCableDelay = channelAddCableDelay.channelAddCableDelay()

def align_and_average_voltage_traces(time_values, voltage_traces):
    """
    Align and plot individual voltage traces and their average.

    Parameters:
    - time_values (numpy array): Array of time values.
    - voltage_traces (numpy array): 2D array where each row represents a voltage trace.

    Returns:
    - average_voltage (numpy array): Averaged and aligned voltage trace.
    """
    # Calculate the cross-correlation between the first trace and the others
    reference_trace = voltage_traces[0]
    aligned_traces = [reference_trace]

    for i in range(1, len(voltage_traces)):
        cross_corr = correlate(reference_trace, voltage_traces[i], mode='full')
        shift = np.argmax(cross_corr) - len(reference_trace) + 1
        aligned_trace = np.roll(voltage_traces[i], shift)
        aligned_traces.append(aligned_trace)

    # Convert the list of aligned traces to a numpy array
    aligned_traces = np.array(aligned_traces)

    # Calculate the mean of the aligned voltage traces along axis 0
    average_voltage = np.mean(aligned_traces, axis=0)
    return average_voltage

def get_bins(bin_width, data):
    """
    Create bins for histogramming data
    Parameters:
    - bin_width (float): Width of each bin
    - data (numpy array): Data to be binned
    
    Returns:
    - bins (numpy array): Array of bin edges    
    """
    min_val = np.min(data)
    max_val = np.max(data)
    num_bins = int((max_val - min_val) / bin_width)
    bins = np.linspace(min_val, max_val, num_bins + 1)
    return bins

def rms_noise(volt_trace):
    """
    Calculate the root mean square (RMS) noise of a voltage trace.
    
    Parameters:
    - volt_trace (numpy array): Voltage trace
    
    Returns:
    - rms (float): RMS noise
    """
    pk_pos = np.argmax(volt_trace)
    noise_wf = np.concatenate((volt_trace[:pk_pos-200], volt_trace[pk_pos+200:]))
    noise = np.sqrt(np.mean(noise_wf**2))

    return noise

def get_snr(volt_trace, ant_type='vpol', scaling=1, atten=0):
    """
    Calculate the signal-to-noise ratio (SNR) of a voltage trace.
    
    Parameters:
    - volt_trace (numpy array): Voltage trace
    - ant_type (str): Antenna type, vpol or hpol, slight correction in snr for hpol, default is 'vpol'
    - scaling (float): Scaling factor for the voltage tracpe to scale non-0 attenuation waveforms to 0dB, default is 1 (scaling on)
    - atten (float): Attenuation in dB, default is 0 (no attenuation)    
    Returns:
    - snr (float): Signal-to-noise ratio
    """
    pos_scale = 1
    neg_scale = 1
    if scaling:
        df = pd.read_csv('/data/user/sanyukta/rno_code/functions/atten_scaling5C.csv')
        if atten in df['atten'].values:
            pos_scale = df['pos_factor'][df['atten']==atten].values[0]
            neg_scale = df['neg_factor'][df['atten']==atten].values[0]
        else:
            print(f'Attenuation {atten} not found in scaling file, no scaling applied')

    vpkpk = np.max(volt_trace)*pos_scale - np.min(volt_trace)*neg_scale
    noise = rms_noise(volt_trace)
    snr = vpkpk/(2*noise)
    if ant_type == 'hpol':
        snr = np.sqrt((vpkpk/(2*noise))**2 - noise**2)

    return snr

def get_hilbert_snr(volt_trace, time_trace, ant_type='vpol', scaling=1, atten=0):
    """
    Calculate the signal-to-noise ratio (SNR) of a voltage trace using the Hilbert transform.
    
    Parameters:
    - volt_trace (numpy array): Voltage trace
    - ant_type (str): Antenna type, vpol or hpol, slight correction in snr for hpol, default is 'vpol'
    - scaling (float): Scaling factor for the voltage tracpe to scale non-0 attenuation waveforms to 0dB, default is 1 (scaling on)
    - atten (float): Attenuation in dB, default is 0 (no attenuation)    
    Returns:
    - snr (float): Signal-to-noise ratio
    """
    h = np.abs(hilbert(volt_trace))
    pk = np.argmax(h)
    integral = simpson(h[pk-60:pk+60], x = time_trace[pk-60:pk+60])
    noise = rms_noise(volt_trace)

    return integral/noise

def set_plot(nrows, ncols, xlabel, ylabel, dpi=200, figsize=None, grid=True):
    """
    Set the plot and labels
    
    Parameters:
    - nrows (int): number of rows in the plot
    - ncols (int): number of columns in the plot
    - dpi (int): dots per inch for the plot, default is 200
    - figsize (tuple): figure size in inches (width, height), default is None
    - xlabel (str): Label for the x-axis
    - ylabel (str): Label for the y-axis
    - grid (bool): Whether to show grid lines, default is True
    Returns:

    """
    fig, ax = plt.subplots(nrows, ncols, dpi=200, figsize=figsize)
    if nrows == 1 and ncols == 1:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if grid:
            plt.grid()
    elif nrows>1 and ncols==1:
        for i in range(nrows):
            ax[i].set_xlabel(xlabel[i])
            ax[i].set_ylabel(ylabel[i])
            if grid:
                ax[i].grid()
    return fig, ax