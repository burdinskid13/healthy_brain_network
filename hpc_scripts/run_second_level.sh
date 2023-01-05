#!/bin/bash
# Job name:
#SBATCH --job-name=second_level
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
#SBATCH --time=02:00:00 # 
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

## Command(s) to run:
module load openmind/anaconda/3-2022.05 # load python module
source ~/.bash_profile # set paths
source $(pipenv --venv)/bin/activate # activate virtual environment

username=maedbh

# navigate to scripts directory
cd /om2/user/${username}/healthy_brain_network/hbn/scripts

# run workflow on model specs created in command above
python3 run_phenotype_secondlevel.py

