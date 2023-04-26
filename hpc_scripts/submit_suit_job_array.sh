#!/bin/bash
subjs=($@)

#base=/nese/mit/group/sig/projects/hbn/hbn_bids # PUT YOUR BIDS DIRECTORY HERE
base=/nese/mit/group/sig/projects/hbn/hbn_bids/derivatives # PUT YOUR BASE DIRECTORY HERE
code=/om2/user/maedbh/healthy_brain_network/hpc_scripts # WHERE CODE IS SAVED

# Get subject names from the directory
if [[ $# -eq 0 ]]; then
    pushd $base
    subjs=($(ls sub-* -d))
    popd
fi

# take the length of the array
# this will be useful for indexing later
len=$(expr ${#subjs[@]} - 1) # len - 1

echo Spawning ${#subjs[@]} sub-jobs.

sbatch --array=0-$len $code/ss_suit.sh $base ${subjs[@]} $code