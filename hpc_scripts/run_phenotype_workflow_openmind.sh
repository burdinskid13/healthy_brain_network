#!/bin/bash
# Job name:
#SBATCH --job-name=workflow_phenotype_hbn
#
# Partition:
#SBATCH --partition=gablab
#
# Nodes:
# SBATCH -N 1 # one node
#
# Cores:
#SBATCH -n 1 # one CPU (hyperthreaded) cores
#
# Wall clock limit:
#SBATCH --time=03:00:00 # 3 hours
# 
# Email Updates:
# SBATCH --mail-user=maedbh@mit.edu
# SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

## Command(s) to run:
module load openmind/anaconda/3-2022.05 # load python module
source ~/.bash_profile # set paths
source $(pipenv --venv)/bin/activate # activate virtual environment

# scripts are stored here:
cd /om2/user/maedbh/healthy_brain_network/hbn/scripts

# run workflow
# python3 run_phenotype_workflow.py --feature-specs --model-specs --run-models-first --run-models-second

# test workflow
python3 test_workflow.py --spec_file=classifier-Child_Measures-Cognitive_Testing-all-DX_01_Cat_binarize.json --cachedir=/home/maedbh/.cache/pydra-ml/cache-wf/
