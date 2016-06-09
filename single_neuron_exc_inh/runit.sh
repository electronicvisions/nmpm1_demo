#!/usr/bin/bash

module purge
module load nmpm_calib_debug/2016-06-09_kljohann_placement

WAFER=4
HICANN=145

for ii in 0 17; do
    cube_sbatch --wafer "$WAFER" --hicann "$HICANN"  -- python measure_single_neuron.py default_parameters.yaml --skip-neurons $ii
done
