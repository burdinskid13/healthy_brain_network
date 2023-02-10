#!/bin/bash
# Job name:
#SBATCH --job-name=continue_workflow
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
#SBATCH --mem=10G
#
# Wall clock limit:
#SBATCH --time=60:00:00 # 
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

## Command(s) to run:
module load openmind/anaconda/3-2022.05 # load python module
source ~/.bash_profile # set paths
source $(pipenv --venv)/bin/activate # activate virtual environment

username=maedbh

echo $("running continue workflow")

# get timestamps
base_dir=/om2/user/${username}/healthy_brain_network/model_specs
TIMESTAMPS=$(ls -d1 "${base_dir}/"*2023*)

for ((m=0; m<${#TIMESTAMPS[@]}; m++)); do \

# loop over workflows and check which ones need to be continued
cd /om2/user/${username}/healthy_brain_network/hbn/scripts
python3 run_phenotype_models.py --spec_dir=${TIMESTAMPS[m]} --cachedir=/om2/user/${username}/bin/.cache/pydra-ml/cache-wf/; done

# delete pydra-ml cache from openmind (takes up to omuch space)
#python3 delete_cache.py --cachedir=/om2/user/${username}/bin/.cache/pydra-ml/cache-wf/
