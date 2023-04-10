#!/bin/bash
# Job name:
#SBATCH --job-name=evaluation
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
#SBATCH --mem=2G
#
# Wall clock limit:
#SBATCH --time=03:00:00
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

## Command(s) to run:
username=maedbh

module load openmind/anaconda/3-2022.05 # load python module
source ~/.bash_profile # set paths
#source $(pipenv --venv)/bin/activate # activate virtual environment
source /om2/user/${username}/bin/miniconda3/bin/activate healthy-brain-network 

echo $"run workflow evaluation"

cd /om2/user/${username}/healthy_brain_network/hbn/scripts

model_dir=/om2/user/${username}/hbn_data/interim/models/all_feature_models/2023-04-01_18-18-41-72/model_395065402/out-localspec-20230402T014245.167362
python3 eval_phenotype_models.py --results_dir=${model_dir} --participant_specs="['participant_spec2.json', 'participant_spec3.json', 'participant_spec4.json', 'participant_spec5.json', 'participant_spec6.json']"