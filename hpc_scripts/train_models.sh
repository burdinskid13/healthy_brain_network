#!/usr/bin/env bash

# Job name:
#SBATCH --job-name=run_phenotypic_models
#
# Partition:
#SBATCH --partition=mit_normal
#
# Nodes:
#SBATCH -N 1 # one node
#SBATCH --exclude=node[028-029,101,142,145,279]
#
# Tasks:
#SBATCH -c 1 # was 16 hyperthreaded cores 
#
# Memory:
#SBATCH --mem=5G
#
# Wall clock limit:
#SBATCH --time=02:00:00
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

# Import arguments from job submission script
args=($@)
participants=(${args[@]:1})
base_dir=$1
model_dir=(${args[-6]})
data_dir=(${args[-5]})
spec_dir=(${args[-4]})
target=(${args[-3]})
feature=(${args[-2]})
pydraml=(${args[-1]})

### SET DIRECTORIES - YOU MAY HAVE TO CHANGE VIRTUAL ENVIRONMENT PATH###
source ~/.bash_profile # set paths
source ~/.bashrc # set paths
source /om2/user/$(whoami)/bin/miniconda3/bin/activate healthy-brain-network

# test -- write out output
# set -x
# env | sort

set -eu # Stop on errors

# index slurm array to grab participant specs
participant=${participants[${SLURM_ARRAY_TASK_ID}]}

echo "${feature}, ${target}, ${pydraml}, ${participant},will be saved to: ${model_dir}"

echo "specs are saved in ${spec_dir}, data are saved in ${data_dir}"
echo "base directory is: ${base_dir}"

# Define scratch directory
scratch=/om2/scratch/tmp/$(whoami)/HBN_Models/ # assign working directory
export SUBJECT_SPEC_DIR=$scratch

# Define python scripts
python_scripts=$base_dir/hbn/scripts

# make specs
python3 $python_scripts/make_specs.py

echo $"run phenotypic models"

# make timestamp for this workflow
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RANDOM_NUMBER=$((1 + $RANDOM % 100))
scratch_dir=$scratch/$TIMESTAMP-$RANDOM_NUMBER
if [ ! -d $scratch_dir ]; then
  mkdir -p $scratch_dir;
fi

# train model
python3 $python_scripts/train_model.py \
--participant_spec=$spec_dir/$participant \
--feature_spec=$spec_dir/$feature \
--target_spec=$spec_dir/$target \
--pydraml_spec=$spec_dir/$pydraml \
--data_dir=$data_dir \
--model_dir=$scratch_dir \
--cache_dir=$scratch/.cache/pydra-ml/cache-wf/ 

# copy model files back
cp -nr $scratch_dir $model_dir

echo "Copied data from ${scratch_dir} to ${model_dir}"