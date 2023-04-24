#!/bin/bash
# Job name:
#SBATCH --job-name=run_suit
#
# Partition:
#SBATCH --partition=gablab
#
# Nodes:
#SBATCH -N 1 # one node
#
# Tasks:
#SBATCH -c 16 # 16 hyperthreaded cores 
#
# Memory:
#SBATCH --mem=15G
#
# Wall clock limit:
#SBATCH --time=10:00:00 # 
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

## Command(s) to run:
module load openmind/matlab
module load openmind/anaconda/3-2022.05 # load python module
source ~/.bash_profile # set paths
source $(pipenv --venv)/bin/activate # activate virtual environment

# set OpenMind username
username=maedbh

# navigate to script directory
cd /om2/user/${username}/healthy_brain_network/hbn/scripts

# set derivatives directory
base_dir=/om2/user/${username}/hbn_data/interim/derivatives

# set spm dir
spm_dir=/om2/user/maedbh/bin/spm12

# inputs to suit
suit_arr=()

# run suit for a given participant
for ((i=0; i<${#suit_arr[@]}; i++)); do \
if [ ! -d ${suit_derivatives}/${suit_arr[i]} ]; then
    matlab -nodisplay -r "run_suit(\"SUIT:run_normalization\", \"${suit_arr[i]}\", \"${space_label}\", \"${base_dir}\", \"${spm_dir}\"); quit;"
fi; done