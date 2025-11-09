import sys, os, argparse
from NuRadioReco.utilities.trace_utilities import *
sys.path.append(os.path.abspath('/data/user/sanyukta/rno_code'))
from reading.data_reading import *
from functions.functions import *
import warnings
warnings.filterwarnings("ignore")
mattak_kwargs = {
    "backend" : "uproot",
    # "backend" : "pyroot"
}

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True, help='Path to the root files to process')
parser.add_argument('--summary', type=bool, required=False, help='Toggle print summary of events processed', default=False)
args = parser.parse_args()
file = args.input_path

savefile = 1  # set to 1 to save output nur file
output_filename = file.split('/')[-1].split('.root')[0]
output = os.path.join('/data/user/sanyukta/rno_code/triboelectric/nur_files', output_filename)
selectors = [lambda event_info : event_info.triggerType != "FORCE"] #remove forced trigger events

reader = basic_read_root(file, selectors=selectors, mattak_kwargs=mattak_kwargs)
print(f"Processing file: {file}, output will be saved to: {output}")
events, eve_times, times, volts = get_eventsvoltstraces(reader, savefile=savefile, filename=output)

if args.summary:
    print(f"{output_filename} has {len(events)} events starting from {eve_times[0]} to {eve_times[-1]}")