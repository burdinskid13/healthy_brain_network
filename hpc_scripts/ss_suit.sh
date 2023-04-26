#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --mem=2GB
#SBATCH --cpus-per-task=8
#SBATCH -J suit

module load openmind/matlab
#source ~/.bash_profile # set paths

set -eu # Stop on errors

##### CHANGE THESE VARIABLES AS NEEDED ######
version=3.15
whoami=maedbh # username

#####
# Import arguments from job submission script
args=($@)
subjs=(${args[@]:1})
base_dir=$1
code_dir=$3

# index slurm array to grab subject
subject=${subjs[${SLURM_ARRAY_TASK_ID}]}

# set fmriprep dir
fmriprepdir=$base_dir/fmriprep_23.0.0/

# Define scratch directory
scratch=/om2/scratch/tmp/$whoami/HBN_SUIT/$subject/ # assign working directory $(whoami)
export SUITENV_SUBJECTS_DIR=$scratch

# Copy anatomicals to scratch directory
pushd $fmriprepdir/$subject
session=$(ls -d ses-*)
t1_file=$(ls -d ses-*/*anat/*VNavNorm_desc-preproc_T1w.nii.gz)
t2_file=$(ls -d ses-*/*anat/*VNavNorm_desc-preproc_T2w.nii.gz)
fname="${subject}_${session}_acq-VNavNorm_space-SUIT_desc-preproc"
cp -nL $fmriprepdir/$subject/$t1_file $scratch/"${fname}_T1w.nii.gz"

# Account for presence of T2
if [ -e $fmriprepdir/$subject/$t2_file ]; then
cp -nL $fmriprepdir/$subject/$t2_file $scratch/"${fname}_T2w.nii.gz"
t2_cmd_text="$scratch/${fname}_T2w.nii.gz"
else t2_cmd_text=''
fi
popd

# Define the command
pushd $scratch
cmd="matlab -nodisplay -r "$code_dir/run_suit(\"SUIT:run_normalization\", \"$scratch/${fname}_T1w.nii.gz\", \"$t2_cmd_text\", \"/om2/user/maedbh/bin/spm12\"); quit;"""

# Run the command
echo "Submitted job for: ${subject}"
echo "$'Command :\n'${cmd}"
${cmd}

# assign output directory
output_dir=${base_dir}/suit_${version}/$subject/$session/anat/

# copy files back
mkdir -p $output_dir
cp -nr $scratch $output_dir
popd