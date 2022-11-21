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
#SBATCH --time=05:00:00 # 5 hours
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

## Command(s) to run:
module load openmind/anaconda/3-2022.05 # load python module
source ~/.bash_profile # set paths
source $(pipenv --venv)/bin/activate # activate virtual environment

cd /om2/user/maedbh/healthy_brain_network/hbn/scripts

# preprocess
python3 preprocess_phenotype.py

# make features
python3 make_phenotype_features.py

# make model specs
python3 make_phenotype_models.py

# run workflow
python3 run_phenotype_models.py --model_spec --cachedir=/om2/users/maedbh/bin/.cache/pydra-ml/cache-wf/
