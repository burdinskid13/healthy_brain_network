#!/bin/bash
# Job name:
#SBATCH --job-name=21_workflow
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
#SBATCH --mem=5G
#
# Wall clock limit:
#SBATCH --time=2-00:00:00
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

echo $"run workflow 21"

cd /om2/user/${username}/healthy_brain_network/hbn/scripts

# preprocess
#python3 preprocess_phenotype.py

# # make features
#python3 make_phenotype_specs.py

# make timestamp for this workflow
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S-%SS)
RANDOM_NUMBER=$((1 + $RANDOM % 100))
spec_dir=/om2/user/${username}/healthy_brain_network/model_specs/${TIMESTAMP}-${RANDOM_NUMBER}

# make model specs
python3 make_phenotype_models.py \
--out_dir=${spec_dir} \
--pydraml_spec=pydraml_spec2.json \
--features="['features-all-all-all-all-spec.json']" \
--target=target_DX_01_Cat_new_binarize-spec.json \
--participant_specs="['participant_spec38.json', 'participant_spec39.json', 'participant_spec40.json', 'participant_spec41.json', 'participant_spec42.json', 'participant_spec43.json', 'participant_spec44.json', 'participant_spec45.json', 'participant_spec46.json']"

# run workflow
python3 run_phenotype_models.py --spec_dir=${spec_dir} --cachedir=/om2/user/${username}/bin/.cache/pydra-ml/cache-wf/

# delete pydra-ml cache from openmind (takes up to omuch space)
#python3 delete_cache.py --cachedir=/om2/user/${username}/bin/.cache/pydra-ml/cache-wf/
