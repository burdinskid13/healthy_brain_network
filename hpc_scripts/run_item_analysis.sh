#!/bin/bash
# Job name:
#SBATCH --job-name=item_analysis
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
#SBATCH --mem=5G
#
# Wall clock limit:
#SBATCH --time=05:00:00 # 
# 
# Email Updates:
#SBATCH --mail-user=maedbh@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT

## Command(s) to run:
module load openmind/anaconda/3-2022.05 # load python module
source ~/.bash_profile # set paths
source $(pipenv --venv)/bin/activate # activate virtual environment

username=maedbh

# navigate to scripts directory
cd /om2/user/${username}/healthy_brain_network/hbn/scripts

# run workflow on model specs created in command above
python3 run_item_analysis.py --split_sex=False --split_age=True --diagnoses="['ADHD', 'No_Diagnosis_Given', 'Anxiety_Disorders', 'Depressive_Disorders', 'Autism_Spectrum_Disorder', 'Specific_Learning_Disorder_with_Impairment_in_Reading']"

python3 run_item_analysis.py --split_sex=False --split_age=False --diagnoses="['ADHD', 'No_Diagnosis_Given', 'Anxiety_Disorders', 'Depressive_Disorders', 'Autism_Spectrum_Disorder', 'Specific_Learning_Disorder_with_Impairment_in_Reading']"

python3 run_item_analysis.py --split_sex=True --split_age=False --diagnoses="['ADHD', 'No_Diagnosis_Given', 'Anxiety_Disorders', 'Depressive_Disorders', 'Autism_Spectrum_Disorder', 'Specific_Learning_Disorder_with_Impairment_in_Reading']"

python3 run_item_analysis.py --split_sex=True --split_age=True --diagnoses="['ADHD', 'No_Diagnosis_Given', 'Anxiety_Disorders', 'Depressive_Disorders', 'Autism_Spectrum_Disorder', 'Specific_Learning_Disorder_with_Impairment_in_Reading']"

python3 run_item_analysis.py --split_sex=True --split_age=False --diagnoses="['ADHD-Combined_Type', 'Hyperactive_Impulsive_Type', 'Inattentive_Type', 'Other_Specified_Attention-Deficit_Hyperactivity_Disorder', 'No_Diagnosis_Given']"

python3 run_item_analysis.py --split_sex=True --split_age=True --diagnoses="['ADHD-Combined_Type', 'Hyperactive_Impulsive_Type', 'Inattentive_Type', 'Other_Specified_Attention-Deficit_Hyperactivity_Disorder', 'No_Diagnosis_Given']"
