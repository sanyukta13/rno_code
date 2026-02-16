from NuRadioReco.modules.io import eventReader
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import stationParameters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import sys, os, argparse
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
from get_summit_weather_data.weather_reader import WeatherData

wd = WeatherData(years=[2025], use_official=True, use_unofficial=True)

reader = eventReader.eventReader()
input_dir = 'nur_files/'
files = [f for f in os.listdir(input_dir) if f.endswith('.nur')]
files.sort()

nurs = defaultdict(list)
for file in files:
    nurs[int(file.split('_')[0].replace('station', ''))].append(file)
nurs = {k: sorted(v) for k, v in nurs.items()}
plt_chs = [13, 16, 19]

for station_id in nurs.keys():
    # Create polar plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    wind_speeds = []
    azimuths = []
    zeniths = []
    
    for file in nurs[station_id]:
        filepath = os.path.join(input_dir, file)
        if not os.path.isfile(filepath):
            continue
        print(f'Processing {filepath}...')
        reader.begin(filepath)
        
        for event in reader.run():
            station = event.get_station(event.get_station_ids()[0])
            station_time = station.get_station_time().unix
            ws = wd.getWindSpeed(station.get_station_time())
            
            try:
                zenith = station[stationParameters.zenith]
                azimuth = station[stationParameters.azimuth]
                print(f'Zenith: {zenith}, Azimuth: {azimuth}, Wind Speed: {ws}')
                wind_speeds.append(ws)
                azimuths.append(azimuth)
                zeniths.append(zenith)
            except:
                continue
    
    if len(azimuths) == 0:
        print(f"No valid data for station {station_id}")
        plt.close()
        continue
    
    # Normalize wind speed for colormap
    norm = Normalize(vmin=min(wind_speeds), vmax=max(wind_speeds))
    cmap = plt.cm.coolwarm
    
    # Plot on polar axes: azimuth as angle, zenith as radius
    scatter = ax.scatter(azimuths, np.rad2deg(zeniths), 
                        c=wind_speeds, cmap=cmap, norm=norm,
                        s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    
    # Configure polar plot
    ax.set_theta_zero_location('N')  # 0° at top (North)
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_ylim(0, 90)  # Zenith from 0° (vertical) to 90° (horizon)
    # ax.set_ylabel('Zenith Angle (degrees)', labelpad=30)
    ax.set_title(f'Station {station_id} - Reconstructed source', 
                 fontsize=14, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Wind Speed (m/s)', fontsize=12)

    plt.savefig(f'plots/reco/station{station_id}_skymap.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f'Saved station{station_id}_skymap.png')