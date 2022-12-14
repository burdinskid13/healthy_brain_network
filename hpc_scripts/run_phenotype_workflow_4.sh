#!/bin/bash
# Job name:
#SBATCH --job-name=2_workflow
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
module load openmind/anaconda/3-2022.05 # load python module
source ~/.bash_profile # set paths
source $(pipenv --venv)/bin/activate # activate virtual environment

username=maedbh

# navigate to scripts directory
cd /om2/user/${username}/healthy_brain_network/hbn/scripts

# preprocess
#python3 preprocess_phenotype.py

# # make features
python3 make_phenotype_specs.py

# make timestamp for this workflow
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
spec_dir=/om2/user/${username}/healthy_brain_network/model_specs/${TIMESTAMP}

# make model specs
python3 make_phenotype_models.py --out_dir=${spec_dir} --pydraml_spec=pydraml_spec2.json --target=target_Sex_binarize-spec.json --participants="['train_participants-No_Diagnosis_Given.csv']"

# run workflow on model specs created in command above
python3 run_phenotype_models.py --spec_dir=${spec_dir} --cachedir=/om2/user/${username}/bin/.cache/pydra-ml/cache-wf/

# delete pydra-ml cache from openmind (takes up to omuch space)
#python3 delete_cache.py --cachedir=/om2/user/${username}/bin/.cache/pydra-ml/cache-wf/
