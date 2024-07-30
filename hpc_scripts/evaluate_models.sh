#!/usr/bin/env bash

# Job name:
#SBATCH --job-name=test_models
#
# Partition:
#SBATCH --partition=normal
#
# Nodes:
#SBATCH -N 1 # one node
#SBATCH --exclude=node[041]
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

### SET DIRECTORIES - YOU MAY HAVE TO CHANGE VIRTUAL ENVIRONMENT PATH###
source ~/.bash_profile # set paths
source ~/.bashrc # set paths
source /om2/user/$(whoami)/bin/miniconda3/bin/activate healthy-brain-network

# Define python scripts
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
python_scripts=$base/hbn/scripts

# Define directories
dirs=(2024-03-04_16-08-04-21 2024-03-04_16-08-07-62 2024-03-04_16-08-11-13 \
2024-03-04_16-08-11-62 2024-03-04_16-08-12-34 2024-03-04_16-08-12-85 \
2024-03-04_16-08-15-45 2024-03-04_16-09-21-51 2024-03-04_16-09-42-26 \
2024-03-05_16-16-39-27)

model_dir=/om2/user/maedbh/hbn_data/interim/models/reading_march/

process_directory() {
  # Get the directory name
  dirn="$1"

  # Define the path to the python script
  python_script="$python_scripts/test_model.py"

  # Define the path to the model spec file (assuming it's inside the first-level directory)
  model_spec_file="$dirn/model_spec-test.json"

  # Execute the python script with arguments
  python3 "$python_script" \
    --model_dir="$dirn" \
    --model_spec="$model_spec_file"

  # (Optional) Add any additional processing within the function for each directory
  echo "Processed directory: $model_dir"
}

# Loop over the directories in the array "dirs"
for model in "${dirs[@]}"; do
  process_directory "$model_dir/$model"
  process_directory "$model_dir/$model_secondlevel"
done
 