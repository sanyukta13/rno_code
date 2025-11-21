# script to plot basic variables from channel parameters against wind speed and time
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

def set_fig(n):
    """Smart subplot layout: returns fig, axes array (always 1D for easy iteration)"""
    layouts = {1: (1,1), 2: (2,1), 3: (3,1), 4: (2,2), 5: (3,2), 6: (3,2), 7: (3,3), 8: (3,3), 9: (3,3)}
    nrows, ncols = layouts.get(n, (int(np.ceil(n/3)), 3))
    fig, ax = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), squeeze=False, sharex=True)
    return fig, ax.flatten()

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--channels', nargs='+', type=int, required=True)
parser.add_argument('--thresh', type=float, default=0.2, help='Minimum maximum amplitude to consider event')
args = parser.parse_args()

input_dir, output_dir, plt_chs, thresh = args.input_dir, args.output_dir, args.channels, args.thresh
os.makedirs(output_dir, exist_ok=True)
print(f'Input Dir: {input_dir}, Output Dir: {output_dir}, Channels: {plt_chs}, Threshold: {thresh}V')
wd = WeatherData(years=[2025], use_official=True, use_unofficial=True)

for file in os.listdir(input_dir):
    if not file.endswith('.nur'):
        continue

    filepath = os.path.join(input_dir, file)
    reader = eventReader.eventReader()
    reader.begin(filepath)

    # Collect all data first to get time range for colormap
    data = {ch: {'ws': [], 'p2p': [], 'max_amp': [], 'max_amp_env': [], 'noise': [], 'time': []} for ch in plt_chs}

    for event in reader.run():
        station = event.get_station(event.get_station_ids()[0])
        station_time = station.get_station_time().unix
        ws = wd.getWindSpeed(station.get_station_time())

        for ch in plt_chs:
            if station.has_channel(ch):
                channel = station.get_channel(ch)
                if channel[chp.maximum_amplitude] < thresh:
                    continue
                data[ch]['ws'].append(ws)
                data[ch]['p2p'].append(channel[chp.P2P_amplitude]/2.0)
                data[ch]['max_amp'].append(channel[chp.maximum_amplitude])
                data[ch]['max_amp_env'].append(channel[chp.maximum_amplitude_envelope])
                data[ch]['noise'].append(channel[chp.noise_rms])
                data[ch]['time'].append(station_time)

    # Get wind speed range for normalization
    try:
        all_ws = np.concatenate([data[ch]['ws'] for ch in plt_chs if data[ch]['ws']])
    except ValueError:
        continue

    if len(all_ws) == 0:
        continue

    ws_min, ws_max = all_ws.min(), all_ws.max()
    norm = Normalize(vmin=ws_min, vmax=ws_max)
    cmap = plt.cm.viridis

    # Create figure and plot
    fig, ax = set_fig(len(plt_chs))
    
    # Create custom legend elements
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, 
               markeredgecolor='black', label='P2P Amplitude/2'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=10, 
               markeredgecolor='black', label='Max Amplitude'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, 
               markeredgecolor='black', label='Max Amp Envelope'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, 
               markeredgecolor='red', label='Noise RMS')
    ]
    
    for i, ch in enumerate(sorted(plt_chs)):
        if not data[ch]['time']:
            continue

        ax[i].scatter(data[ch]['time'], data[ch]['p2p'], c=data[ch]['ws'], cmap=cmap, norm=norm, marker='o',
                     s=30, edgecolors='black', linewidths=0.5)
        ax[i].scatter(data[ch]['time'], data[ch]['max_amp'], c=data[ch]['ws'], cmap=cmap, norm=norm, marker='*',
                     s=50, edgecolors='black', linewidths=0.5)
        ax[i].scatter(data[ch]['time'], data[ch]['max_amp_env'], c=data[ch]['ws'], cmap=cmap, norm=norm, marker='s',
                     s=30, edgecolors='black', linewidths=0.5)
        ax[i].scatter(data[ch]['time'], data[ch]['noise'], c=data[ch]['ws'], cmap=cmap, norm=norm, marker='^',
                     s=20, edgecolors='red', linewidths=0.5)
        
        ax[i].set_title(f'Channel {ch}', fontsize=12)
        ax[i].set_ylabel('Amplitude (V)', fontsize=11)
        ax[i].grid(True, alpha=0.3)
        ax[i].legend(handles=legend_elements, fontsize=8, fancybox=True, framealpha=0.5)
        ax[i].tick_params(axis='x', rotation=45)

        # Only show x-label on bottom row
        if i >= len(plt_chs) - (len(plt_chs) % 3 if len(plt_chs) > 3 else len(plt_chs)):
            ax[i].set_xlabel('Time', fontsize=11)

    # Hide extra subplots
    for idx in range(len(plt_chs), len(ax)):
        ax[idx].axis('off')

    # Add single colorbar for all subplots
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label('Wind Speed (m/s)', fontsize=11)

    plt.savefig(os.path.join(output_dir, f'{file[:-13]}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {file}')

