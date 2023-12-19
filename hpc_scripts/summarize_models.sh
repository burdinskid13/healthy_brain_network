#!/bin/bash
# Job name:
#SBATCH --job-name=summarize_models
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

echo "overall model summary will be saved to: ${model_dir}"

# Define python scripts
python_scripts=$base_dir/hbn/scripts

# run workflow
cmd="python3 $python_scripts/summarize_models.py --model_dir=$model_dir

# Run the command
echo "$'Command :\n'${cmd}"
${cmd}