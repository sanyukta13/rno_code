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
os.makedirs(output_dir, exist_ok=True)
if plt_chs == [13, 16, 19]:
    orientations = [30, 150, 90]
else:
    orientations = np.zeros(len(plt_chs))
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
        fig = plt.figure(figsize=(14, 6))
        ax0 = fig.add_subplot(1, 2, 1)  
        ax1 = fig.add_subplot(1, 2, 2, projection='polar')  
        tot_evs = 0; zeniths = {}; azimuths = {}; times = {}; scales = {}; ws = {}; snr = {}
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
                azimuths[event.get_id()] = azimuth
                times[event.get_id()] = station_time
                # corr = calc_dispersivity(channel.get_hilbert_envelope()) 
                ws[event.get_id()] = wd.getWindSpeed(station.get_station_time())
                scales[event.get_id()] = scale
                snr[event.get_id()] = signal_noise

        print(f'Total events processed: {tot_evs}, Events with zenith/azimuth: {len(zeniths)}')
        if len(zeniths) != 0:
            # ax0_wind = ax0.twinx()
            # ax0_wind.plot(times.values(), ws.values(), color='grey', linestyle='--', label='Wind Speed (m/s)')
            # ax0_wind.scatter(times.values(), ws.values(), color='salmon', marker='x')
            # ax0_wind.set_ylabel('Wind Speed (m/s)')
            scatter = ax0.scatter(times.values(), scales.values(), c=ws.values(), cmap='viridis', s=[5*v for v in snr.values()], edgecolors='midnightblue')
            ax0.set_xlabel('Time (unix)')
            ax0.set_ylabel('Landau Scale Parameter')
            ax0.grid()
            start_time = datetime.fromtimestamp(list(times.values())[0], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            end_time = datetime.fromtimestamp(list(times.values())[-1], tz=timezone.utc).strftime('%H:%M')
            ax0.legend(title=f'St {station.get_id()} ch{ch} {ori}\u00b0 CCW E\n{start_time} - {end_time}', framealpha=0.5)

            scatter_polar = ax1.scatter(azimuths.values(), zeniths.values(), c=ws.values(), 
                                        cmap='viridis', s=[5*v for v in snr.values()],
                                        edgecolors='midnightblue', linewidths=0.5)
            ax1.vlines(np.deg2rad(ori), 0, 90, color='maroon', linestyle='--')
            ax1.vlines(np.deg2rad(ori+90), 0, 90, color='salmon', linestyle='--')
            ax1.vlines(np.deg2rad(ori+270), 0, 90, color='salmon', linestyle='--')

            ax1.set_theta_zero_location('E')  # East at top
            ax1.set_theta_direction(1)  # counter-clockwise
            ax1.set_ylim(0, 90)  # Zenith from 0° to 90°

            fig.colorbar(scatter, ax=[ax1], label='Wind Speed (m/s)')
            plt.savefig(f'{output_dir}/{file.split("_")[0]}{file.split("_")[1]}_ch{ch}-landau.png', bbox_inches='tight')
        plt.close()
