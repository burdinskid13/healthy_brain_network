#!/bin/bash
# Job name:
#SBATCH --job-name=workflow_phenotype_hbn
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
#SBATCH --mem=20G
#
# Wall clock limit:
#SBATCH --time=8:00:00 # 5 hours
# 
# Email Updates:
#SBATCH --mail-user=shreyark@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

## Command(s) to run:
module load openmind/anaconda/3-2022.05 # load python module
source ~/.bash_profile # set paths
source $(pipenv --venv)/bin/activate # activate virtual environment

username=shreyark

# scripts are stored here:
cd /om2/user/${username}/healthy_brain_network/hbn/tests

# test workflow
python3 test_workflow.py --cachedir=/om2/user/${username}/bin/.cache/pydra-ml/cache-wf/
