#!/usr/bin/env bash
# Job name:
#SBATCH --job-name=summarize_models
#
# Partition:
#SBATCH --partition=mit_normal
#
# Nodes:
#SBATCH -N 1 # one node
#SBATCH --exclude=node[041]
#
# Tasks:
#SBATCH -c 16 # 16 hyperthreaded cores 
#
# Memory:
#SBATCH --mem=2G
#
# Wall clock limit:
#SBATCH --time=2-00:00:00
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

# Import arguments from job submission script
args=($@)
base_dir=${args[-2]}
model_dir=${args[-1]}

### SET DIRECTORIES - YOU MAY HAVE TO CHANGE VIRTUAL ENVIRONMENT PATH###
source ~/.bash_profile # set paths
source ~/.bashrc # set paths
source /om2/user/$(whoami)/bin/miniconda3/bin/activate healthy-brain-network

echo "overall model summary will be saved to: ${model_dir} and base_dir is ${base_dir}"

# Define python scripts
python_scripts=$base_dir/hbn/scripts

# run workflow
cmd="python3 $python_scripts/summarize_models.py --model_dir=$model_dir"

# Run the command
echo "$'Command :\n'${cmd}"
${cmd}