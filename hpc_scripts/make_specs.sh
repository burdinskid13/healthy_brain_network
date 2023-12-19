#!/bin/bash
# Job name:
#SBATCH --job-name=preprocess_models
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
#SBATCH --time=01:00:00
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

### SET DIRECTORIES ###
source ~/.bash_profile # set paths
source ~/.bashrc # set paths
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
out_dir=$base/model_specs

# activate virtual environment
source /om2/user/$(whoami)/bin/miniconda3/bin/activate healthy-brain-network 

# preprocess phenotypes and make specs
python3 $base/hbn/scripts/make_specs.py --out_dir=$out_dir