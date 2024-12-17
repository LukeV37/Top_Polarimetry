#!/bin/bash

# Extract LHE file with gzip
gzip -dk ./pp_tt_semi_full/Events/run_01/unweighted_events.lhe.gz

# Find line with version number and delete the proceeding warning 
# This is needed since I am using git tag instead of production version
# MadGraph is enshitified... Sorry devs... K.I.S.S.
line_num="$(awk '/VERSION 3.5.5/ {print NR}' pp_tt_semi_full/Events/run_01/unweighted_events.lhe)"
start_line=$((line_num+1))
end_line=$((line_num+4))
sed -i "${start_line},${end_line}d" pp_tt_semi_full/Events/run_01/unweighted_events.lhe

# Now that warning message is removed, use LHEReader.py to convert LHE file to root file
python3 include/LHEReader.py --input pp_tt_semi_full/Events/run_01/unweighted_events.lhe --output partons.root

# Clean workspace (uncompressed version no longer needed)
rm -f ./pp_tt_semi_full/Events/run_01/unweighted_events.lhe
