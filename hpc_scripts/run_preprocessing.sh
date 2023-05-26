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
#SBATCH --mem=2G
#
# Wall clock limit:
#SBATCH --time=02:00:00
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

### SET DIRECTORIES ###
source ~/.bash_profile # set paths
source ~/.bashrc # set paths
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE

# activate virtual environment
source /om2/user/$(whoami)/bin/miniconda3/bin/activate healthy-brain-network 

# preprocess phenotypes and make specs
#python3 $base/hbn/scripts/preprocess_phenotype.py
python3 $base/hbn/scripts/make_phenotype_specs.py