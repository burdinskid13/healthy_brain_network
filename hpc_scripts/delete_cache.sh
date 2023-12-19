#!/bin/bash
# Job name:
#SBATCH --job-name=delete_cache
#
# Partition:
#SBATCH --partition=gablab
#
# Nodes:
#SBATCH -N 1 # one node
#
# Tasks:
#SBATCH -c 8 # 8 hyperthreaded cores 
#
# Memory:
#SBATCH --mem=1G
#
# Wall clock limit:
#SBATCH --time=2-00:00:00
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

### SET DIRECTORIES ###
source ~/.bash_profile # set paths
source ~/.bashrc # set paths
base_dir=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
python_scripts=$base_dir/hbn/scripts

# activate virtual environment
source /om2/user/$(whoami)/bin/miniconda3/bin/activate healthy-brain-network

# delete pydra-ml cache from openmind (takes up to omuch space)
python3 $python_scripts/delete_cache.py --cachedir=/om2/user/$(whoami)/bin/.cache/pydra-ml/cache-wf/
