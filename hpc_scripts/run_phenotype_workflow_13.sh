#!/bin/bash
# Job name:
#SBATCH --job-name=13_workflow
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
#SBATCH --mem=4G
#
# Wall clock limit:
#SBATCH --time=24:00:00 # 
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

## Command(s) to run:
module load openmind/anaconda/3-2022.05 # load python module
source ~/.bash_profile # set paths
source $(pipenv --venv)/bin/activate # activate virtual environment

username=maedbh

echo $"run workflow 13 - continuation"

cd /om2/user/${username}/healthy_brain_network/hbn/scripts

# make timestamp for this workflow
TIMESTAMP=2023-02-15_13-58-43-43S
spec_dir=/om2/user/${username}/healthy_brain_network/model_specs/${TIMESTAMP}

# run workflow
python3 run_phenotype_models.py --spec_dir=${spec_dir} --cachedir=/om2/user/${username}/bin/.cache/pydra-ml/cache-wf/

# delete pydra-ml cache from openmind (takes up to omuch space)
#python3 delete_cache.py --cachedir=/om2/user/${username}/bin/.cache/pydra-ml/cache-wf/
