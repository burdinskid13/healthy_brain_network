#!/bin/bash
# Job name:
#SBATCH --job-name=transfer_data
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
#SBATCH --mem=1G
#
# Wall clock limit:
#SBATCH --time=02:00:00
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

spec_dir=/om2/scratch/tmp/maedbh/HBN_Models/*
out_dir=/om2/user/maedbh/hbn_data/interim/models/adhd-models

cp -nr $spec_dir $out_dir