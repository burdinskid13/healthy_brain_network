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
#SBATCH --mem=7G
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
out_dir=$3
target=(${args[-1]})

### SET DIRECTORIES - YOU MAY HAVE TO CHANGE VIRTUAL ENVIRONMENT PATH###
source ~/.bash_profile # set paths
source ~/.bashrc # set paths
source /om2/user/$(whoami)/bin/miniconda3/bin/activate healthy-brain-network 

set -eu # Stop on errors

# index slurm array to grab participant specs
spec=${specs[${SLURM_ARRAY_TASK_ID}]}

# Define scratch directory
scratch=/om2/scratch/tmp/$(whoami)/HBN_Models/ # assign working directory
export SUBJECT_SPEC_DIR=$scratch

# Define python scripts
python_scripts=$base_dir/hbn/scripts

echo $"run phenotypic models"

# make timestamp for this workflow
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RANDOM_NUMBER=$((1 + $RANDOM % 100))
spec_dir=$scratch/$TIMESTAMP-$RANDOM_NUMBER

# make model specs
python3 $python_scripts/make_phenotype_models.py \
--out_dir=$spec_dir \
--pydraml_spec=$base_dir/model_specs/pydraml_spec2.json \
--features="['${base_dir}/features/features-all-all-all-all-spec.json']" \
--target=$base_dir/features/$target \
--participant_spec=$base_dir/model_specs/participant_specs/$spec

# run workflow
cmd="python3 $python_scripts/run_phenotype_models.py --spec_dir=$spec_dir --cachedir=$scratch/.cache/pydra-ml/cache-wf/"

# Run the command
echo "Submitted job for: ${spec}"
echo "$'Command :\n'${cmd}"
${cmd}

# copy files back
mkdir -p $out_dir
cp -nr $spec_dir/ $out_dir