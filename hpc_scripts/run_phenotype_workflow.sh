#!/bin/bash
# Job name:
#SBATCH --job-name=workflow_phenotype_hbn
#
# Account:
#SBATCH --account=fc_cerebellum
#
# Partition:
#SBATCH --partition=savio2

# Quality of Service:
#SBATCH --qos=savio_normal
#
# Wall clock limit:
#SBATCH --time=50:00:00

## Command(s) to run:
module load python/3.9.12
source ~/.bash_profile
source $(pipenv --venv)/bin/activate

cd /global/scratch/users/maedbhking/projects/healthy_brain_network/hbn/scripts

# run workflow
python3 run_phenotype_workflow.py --no-feature-specs --model-specs --run-models-first --run-models-second
