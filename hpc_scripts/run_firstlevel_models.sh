#!/bin/bash
# Job name:
#SBATCH --job-name=run_phenotypic_models
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
#SBATCH --time=2-00:00:00
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

# Import arguments from job submission script
args=($@)
specs=(${args[@]:1})
base_dir=$1
model_dir=(${args[-5]})
data_dir=(${args[-4]})
target=(${args[-3]})
feature=(${args[-2]})
pydraml=(${args[-1]})

# ### SET DIRECTORIES - YOU MAY HAVE TO CHANGE VIRTUAL ENVIRONMENT PATH###
# source ~/.bash_profile # set paths
# source ~/.bashrc # set paths
# source /om2/user/$(whoami)/bin/miniconda3/bin/activate healthy-brain-network

set -eu # Stop on errors

# index slurm array to grab participant specs
participant=${specs[${SLURM_ARRAY_TASK_ID}]}

echo "${feature}, ${target}, ${pydraml}, ${participant}will be saved to: ${model_dir}"

# Define scratch directory
scratch=/om2/scratch/tmp/$(whoami)/HBN_Models/ # assign working directory
export SUBJECT_SPEC_DIR=$scratch

# Define python scripts
python_scripts=$base_dir/hbn/scripts

echo $"run phenotypic models"

# make timestamp for this workflow
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RANDOM_NUMBER=$((1 + $RANDOM % 100))
scratch_dir=$scratch/$TIMESTAMP-$RANDOM_NUMBER

# make model specs
python3 $python_scripts/make_firstlevel_model.py \
--feature_spec=$feature \
--target_spec=$target \
--participant_spec=$participant \
--pydraml_spec=$pydraml \
--data_dir=$data_dir \
--out_dir=$scratch_dir

# run workflow
cmd="python3 $python_scripts/run_model.py --model_dir=$scratch_dir --cache_dir=$scratch/.cache/pydra-ml/cache-wf/"

# Run the command
echo "Submitted job for: ${participant}"
echo "$'Command :\n'${cmd}"
${cmd}

# copy files back
cp -nr $scratch_dir $model_dir

echo "$'Copied data from ${scratch_dir} to ${model_dir}"