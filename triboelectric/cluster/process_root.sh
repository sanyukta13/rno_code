#!/bin/bash

export CVMFS_PROFILE=/cvmfs/rnog.opensciencegrid.org/software/
source $CVMFS_PROFILE/setup.sh
export RNOG_VENV=/home/sanyukta/software/rnog_venv/bin/
source $RNOG_VENV/activate

INPUT_DIR="/data/user/sanyukta/rno_data/highwind/"

for file in "$INPUT_DIR"/*/*.root; do
    [ -e "$file" ] || continue
    python /data/user/sanyukta/rno_code/triboelectric/cluster/process_root.py --input_path "$file" --summary >> process_root.log 2>&1
done

echo "Processing complete."