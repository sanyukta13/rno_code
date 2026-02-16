# script to plot the landau scale fit parameter of two channels against each other
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
print(f'Input Dir: {input_dir}, Output Dir: {output_dir}, Channels: {plt_chs}')
wd = WeatherData(years=[2025], use_official=True, use_unofficial=True)

for file in os.listdir(input_dir):
    if not file.endswith('.nur'):
        continue
    filepath = os.path.join(input_dir, file)
    reader = eventReader.eventReader()
    reader.begin(filepath)
    print(f'Processing file: {file}, for chs {plt_chs}')

    fig, ax = plt.subplots(1, 3, figsize=(14, 6))
    tot_evs = 0; zeniths = {}; azimuths = {}; times = {}; scales = {}; ws = {}
    for ch in plt_chs:
        scales[ch] = {}

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
            skip = 0
            for ch in plt_chs:
                channel = station.get_channel(ch)
                scale, fit = fit_landau(channel.get_hilbert_envelope())
                signal_noise = channel[chp.SNR]['peak_2_peak_amplitude_split_noise_rms']
                
                if scale > 500 or signal_noise < 4:   #avoiding weirdly fitted events
                    skip = 1

                scales[ch][event.get_id()] = scale

            if skip: 
                for ch in plt_chs:
                    scales[ch].pop(event.get_id(), None)
                continue

            zeniths[event.get_id()] = np.rad2deg(zenith)
            azimuths[event.get_id()] = azimuth
            times[event.get_id()] = station_time
            # corr = calc_dispersivity(channel.get_hilbert_envelope()) 
            ws[event.get_id()] = wd.getWindSpeed(station.get_station_time())   

    print(f'Total events processed: {tot_evs}, Events with zenith/azimuth: {len(zeniths)}')

    if len(zeniths) != 0:
        for i in range(len(plt_chs)):
            ch1 = plt_chs[i]; ch2 = plt_chs[(i+1)%len(plt_chs)]
            scatter = ax[i].scatter(scales[ch1].values(), scales[ch2].values(), c=ws.values(), cmap='viridis', edgecolors='midnightblue')
            ax[i].set_xlabel(f'Ch{ch1}')
            ax[i].set_ylabel(f'Ch{ch2}')
            ax[i].grid()
            start_time = datetime.fromtimestamp(list(times.values())[0], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            end_time = datetime.fromtimestamp(list(times.values())[-1], tz=timezone.utc).strftime('%H:%M')
        ax[0].legend(title=f'St {station.get_id()}\n{start_time} - {end_time}', framealpha=0.5)
        fig.colorbar(scatter, ax=[ax[2]], label='Wind Speed (m/s)')
        plt.savefig(f'{output_dir}/{file.split("_")[0]}{file.split("_")[1]}_comp.png', bbox_inches='tight')
    plt.close()
