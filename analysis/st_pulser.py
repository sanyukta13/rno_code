import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from NuRadioReco.utilities.fft import freq2time, time2freq, freqs
sys.path.append(os.path.abspath('/data/user/sanyukta/rno_code'))
from reading.data_reading import *
from functions.functions import *

# constants
chs = [0, 1, 2, 3, 8, 4, 5, 6, 7]
CAL_PULSER_THRESHOLD = 200000
selector = [lambda event_info: (event_info.sysclk - event_info.sysclkLastPPS[0]) % (2**32) <= CAL_PULSER_THRESHOLD]
fibers = [0, 1]
atts = [0, 5, 10, 20]

df = pd.DataFrame(columns=['station', 'run', 'fiber', 'att', 'ch', 'snr', 'snr_sigma', 'hilbert_snr', 'hilbert_snr_sigma', 'zenith', 'zenith_sigma',
                           'vpos', 'vpos_sigma', 'vneg', 'vneg_sigma', 'integral', 'integral_sigma', 'vdiff', 'noise', 'noise_sigma', 'hilbert_noise', 'hilbert_noise_sigma'])
def run(st_id):
    station_id = st_id
    for fiber in fibers:
        for att in atts:
            root_files_dir = '/data/user/sanyukta/rno_data/cal_pulser/station'+str(station_id)+'/fiber'+str(fiber)+'/'+str(att)+'dB/'
            root_files = [f for f in os.listdir(root_files_dir) if f.endswith('.root')]
            for root_file in root_files:
                root_dir = os.path.join(root_files_dir, root_file)
                run_id = int(root_file.split('_')[1][3:])
                cpevents, cptimes, cpvolts = get_eventsvoltstraces(basic_read_root(root_dir, selectors = selector), band_pass=1, pulse_filter=1)
                if len(cpevents) == 0:
                    continue
                snr_list = {}; hilbert_snr_list = {}; snr_sigma = {}; hilbert_snr_sigma = {}
                zenith_list = {}; zenith_sigma = {}; vdiff = {}; vpkpos = {}; vpkpos_sigma = {}
                vpkneg = {}; vpkneg_sigma = {}; integral = {}; integral_sigma = {}
                noise_list = {}; noise_sigma = {}; hilbert_noise_list = {}; hilbert_noise_sigma = {}
                x_t, y_t, z_t = get_cp_pos(station_id, run_id)
                for ch in chs:
                    snr_cp = []; hsnr_cp = []
                    vpos = []; vneg = []; integral_cp = []
                    noise = []; hilbert_noise = []
                    if ch==4 or ch==8:
                        for idx,eve in enumerate(cpevents):
                            volt_trace = cpvolts[idx][ch]
                            vneg.append(np.min(volt_trace)); vpos.append(np.max(volt_trace))
                            integral_cp.append(get_hilbert_integral(volt_trace, cptimes[idx][ch]))
                            noise.append(rms_noise(volt_trace, method='peak'))
                            hilbert_noise.append(rms_noise(volt_trace, method='hilbert'))
                            snr_cp.append(get_snr(volt_trace, ant_type='hpol', atten=att))
                            hsnr_cp.append(get_hilbert_snr(volt_trace, cptimes[idx][ch], ant_type='hpol', atten=att))
                    else:
                        for idx,eve in enumerate(cpevents):
                            volt_trace = cpvolts[idx][ch]
                            vneg.append(np.min(volt_trace)); vpos.append(np.max(volt_trace))
                            integral_cp.append(get_hilbert_integral(volt_trace, cptimes[idx][ch]))
                            noise.append(rms_noise(volt_trace, method='peak'))
                            hilbert_noise.append(rms_noise(volt_trace, method='hilbert'))
                            snr_cp.append(get_snr(volt_trace, atten=att))
                            hsnr_cp.append(get_hilbert_snr(volt_trace, cptimes[idx][ch], atten=att))

                    snr_list[ch] = np.mean(snr_cp); hilbert_snr_list[ch] = np.mean(hsnr_cp)
                    snr_sigma[ch] = np.std(snr_cp); hilbert_snr_sigma[ch] = np.std(hsnr_cp)
                    vpkpos[ch] = np.mean(vpos); vpkpos_sigma[ch] = np.std(vpos)
                    vpkneg[ch] = np.mean(vneg); vpkneg_sigma[ch] = np.std(vneg)
                    integral[ch] = np.mean(integral_cp); integral_sigma[ch] = np.std(integral_cp)
                    vdiff[ch] = vpkpos[ch] - np.abs(vpkneg[ch])
                    noise_list[ch] = np.mean(noise); noise_sigma[ch] = np.std(noise)
                    hilbert_noise_list[ch] = np.mean(hilbert_noise); hilbert_noise_sigma[ch] = np.std(hilbert_noise)

                    x_r, y_r, z_r = get_ch_pos(station_id, ch)
                    zenith_list[ch], zenith_sigma[ch] = zenith(x_r, y_r, z_r, x_t, y_t, z_t)
                    
                # Create a new DataFrame to append
                new_data = pd.DataFrame({
                    'station': [station_id] * len(snr_list.keys()),
                    'run': [run_id] * len(snr_list.keys()),
                    'fiber': [fiber] * len(snr_list.keys()),
                    'att': [att] * len(snr_list.keys()),
                    'ch': list(snr_list.keys()),
                    'snr': list(snr_list.values()),
                    'snr_sigma': list(snr_sigma.values()),
                    'hilbert_snr': list(hilbert_snr_list.values()),
                    'hilbert_snr_sigma': list(hilbert_snr_sigma.values()),
                    'zenith': list(zenith_list.values()),
                    'zenith_sigma': list(zenith_sigma.values()),
                    'vpos': list(vpkpos.values()),
                    'vpos_sigma': list(vpkpos_sigma.values()),
                    'vneg': list(vpkneg.values()),
                    'vneg_sigma': list(vpkneg_sigma.values()),
                    'integral': list(integral.values()),
                    'integral_sigma': list(integral_sigma.values()),
                    'vdiff': list(vdiff.values()),
                    'noise': list(noise_list.values()),
                    'noise_sigma': list(noise_sigma.values()),
                    'hilbert_noise': list(hilbert_noise_list.values()),
                    'hilbert_noise_sigma': list(hilbert_noise_sigma.values())
                })

                df = pd.concat([df, new_data], ignore_index=True)

        # save the dataframe to a CSV file
    df.to_csv(f'/data/user/sanyukta/rno_data/sheets/st{station_id}.csv', index=False)

if __name__=="__main__": 
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('station_id', type=int)
    run(parser.parse_args().station_id)