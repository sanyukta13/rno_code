# script to plot the landau scle fit parameter against wind speed and time
from NuRadioReco.modules.io import eventReader
from NuRadioReco.framework.parameters import channelParameters as chp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import sys, os, argparse
import numpy as np
from datetime import datetime, timezone
from get_summit_weather_data.weather_reader import WeatherData
from NuRadioReco.framework.parameters import stationParameters
sys.path.append('../../functions')
from functions import *

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--channels', nargs='+', type=int, required=True)
args = parser.parse_args()

input_dir, output_dir, plt_chs = args.input_dir, args.output_dir, args.channels
if plt_chs == [13, 16, 19]:
    orientations = [30, 150, 90]
else:
    orientations = np.zeroes(len(plt_chs))
os.makedirs(output_dir, exist_ok=True)
print(f'Input Dir: {input_dir}, Output Dir: {output_dir}, Channels: {plt_chs}, Orientations: {orientations}')
wd = WeatherData(years=[2025], use_official=True, use_unofficial=True)

for file in os.listdir(input_dir):
    if not file.endswith('.nur'):
        continue
    filepath = os.path.join(input_dir, file)
    reader = eventReader.eventReader()
    reader.begin(filepath)
    print(f'Processing file: {file}, for chs {plt_chs}')

    for ori, ch in zip(orientations, plt_chs):
        fig, ax = plt.subplots(2, 2, figsize=(14, 14), sharey='row', sharex='col', gridspec_kw={'wspace': 0.02, 'hspace': 0.02})
        tot_evs = 0; zeniths = {}; azimuths = {}; times = {}; scales = {}; ws = {}; snr = {}; noise = {}
        for event in reader.run():
            tot_evs += 1
            station = event.get_station()
            station_time = station.get_station_time().unix

            try:
                zenith = station[stationParameters.zenith]
                azimuth = station[stationParameters.azimuth]
            except: 
                continue

            if zenith:
                channel = station.get_channel(ch)
                scale, fit = fit_landau(channel.get_hilbert_envelope())
                signal_noise = channel[chp.SNR]['peak_2_peak_amplitude_split_noise_rms']
                
                if scale > 500 or signal_noise < 4:   #avoiding weirdly fitted events
                    continue

                zeniths[event.get_id()] = np.rad2deg(zenith)
                az = np.rad2deg(azimuth) - ori 
                azimuths[event.get_id()] = az + 360 if az < 0 else az
                snr[event.get_id()] = signal_noise/np.cos(np.deg2rad(azimuths[event.get_id()]))**2 #correcting for azimuthal gain pattern
                times[event.get_id()] = station_time
                # corr = calc_dispersivity(channel.get_hilbert_envelope()) 
                ws[event.get_id()] = wd.getWindSpeed(station.get_station_time())
                scales[event.get_id()] = scale
                noise[event.get_id()] = round(channel[chp.noise_rms]*1e3,1)
        
        print(f'Total events processed: {tot_evs}, Events with zenith/azimuth: {len(zeniths)}')
        if len(zeniths) != 0:
            scatter = ax[0, 1].scatter(zeniths.values(), scales.values(), c=ws.values(), cmap='viridis', s=[10*v for v in noise.values()], edgecolors='midnightblue')
            scatter = ax[0, 0].scatter(azimuths.values(), scales.values(), c=ws.values(), cmap='viridis', s=[10*v for v in noise.values()], edgecolors='midnightblue')
            scatter = ax[1, 1].scatter(zeniths.values(), snr.values(), c=ws.values(), cmap='viridis', s=[10*v for v in noise.values()], edgecolors='midnightblue')
            scatter = ax[1, 0].scatter(azimuths.values(), snr.values(), c=ws.values(), cmap='viridis', s=[10*v for v in noise.values()], edgecolors='midnightblue')

            for i in range(2):
                ax[i, 0].set_xlim(0, 360)
                ax[i, 0].axvline(x=90, color='orange', ls='--')
                ax[i, 0].axvline(x=270, color='orange', ls='--')
                ax[1,i].set_yscale('log')
                for j in range(2):
                    ax[i, j].grid()
                    ax[i, j].minorticks_on()

            ax[1, 1].set_xlabel(f'Zenith(\u00b0)', fontsize=12)
            ax[1, 0].set_xlabel(f'Azimuth(\u00b0)', fontsize=12)
            ax[0, 0].set_ylabel('Landau Scale Parameter', fontsize=12)
            ax[1, 0].set_ylabel('SNR (corrected for azimuthal gain)', fontsize=12)

            start_time = datetime.fromtimestamp(list(times.values())[0], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            end_time = datetime.fromtimestamp(list(times.values())[-1], tz=timezone.utc).strftime('%H:%M')
            n_mean = np.mean(list(noise.values())); n_std = np.std(list(noise.values()))
            ax[0, 0].legend(title=f'St {station.get_id()} ch{ch} {ori}\u00b0 CCW E\n{start_time} - {end_time}\nNoise: {n_mean:.1f} ± {n_std:.1f} mV', framealpha=0.5)

            fig.colorbar(scatter, ax=[ax[0,1], ax[1,1]], label='Wind Speed (m/s)')
            plt.savefig(f'{output_dir}/{file.split("_")[0]}{file.split("_")[1]}_ch{ch}_thetaphi.png', bbox_inches='tight')
        plt.close()