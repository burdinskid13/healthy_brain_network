#!/bin/bash
# Job name:
#SBATCH --job-name=run_secondlevel_phenotypic_models
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
#SBATCH --mem=7G
#
# Wall clock limit:
#SBATCH --time=10:00:00
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

# Import arguments from job submission script
args=($@)
specs=(${args[@]:1})
base_dir=$1
model_dir=(${args[-2]})
out_dir=(${args[-1]})

echo "secondlevel models will be saved to: ${out_dir}"

### SET DIRECTORIES - YOU MAY HAVE TO CHANGE VIRTUAL ENVIRONMENT PATH###
source ~/.bash_profile # set paths
source ~/.bashrc # set paths
source /om2/user/$(whoami)/bin/miniconda3/bin/activate healthy-brain-network 

set -eu # Stop on errors

# index slurm array to grab specs
spec=${specs[${SLURM_ARRAY_TASK_ID}]}

# Define scratch directory
scratch=/om2/scratch/tmp/$(whoami)/HBN_Models/ # assign working directory
export SUBJECT_SPEC_DIR=$scratch

# Define python scripts
python_scripts=$base_dir/hbn/scripts

echo $"run secondlevel phenotypic models"

# make timestamp for this workflow
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RANDOM_NUMBER=$((1 + $RANDOM % 100))
spec_dir=$scratch/$TIMESTAMP-$RANDOM_NUMBER

mkdir -p $spec_dir

# make new model spec based on feature selection from first-level modeling
python3 $python_scripts/make_secondlevel_features.py --old_dir=$model_dir/$spec --new_dir=$spec_dir 

# run workflow
cmd="python3 $python_scripts/run_phenotype_models.py --spec_dir=$spec_dir --cachedir=$scratch/.cache/pydra-ml/cache-wf/"

# Run the command
echo "Submitted job for: ${spec}"
echo "$'Command :\n'${cmd}"
${cmd}

# copy files back
cp -nr $spec_dir $out_dir

echo "$'Copied data from ${spec_dir} to ${out_dir}"