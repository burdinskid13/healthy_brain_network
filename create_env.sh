#!/bin/bash

# Job name:
#SBATCH --job-name=create_env
#
# Partition:
#SBATCH --partition=mit_normal
#
# Nodes:
#SBATCH -N 1 # one node
#
# Tasks:
#SBATCH -c 1 # was 16 hyperthreaded cores
#
# Memory:
#SBATCH --mem=20G
#
# Wall clock limit:
#SBATCH --time=03:00:00
#
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

### SET DIRECTORIES - YOU MAY HAVE TO CHANGE VIRTUAL ENVIRONMENT PATH###
source ~/.bash_profile # set paths
source ~/.bashrc # set paths

# create environment
conda env create -f /orcd/data/satra/001/users/maedbh/healthy_brain_network/environment.yml

# activate environment
source /orcd/data/satra/001/users/maedbh/bin/miniconda3/bin/activate healthy-brain-network

# add editable packages
pip install -e . # install src package
pip install -e /orcd/data/satra/001/users/maedbh/pydra-ml # install pydra ml development package
