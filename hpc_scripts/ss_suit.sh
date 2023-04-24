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

#####
# Import arguments from job submission script
args=($@)
subjs=(${args[@]:1})
anat_dir=$1
code_dir=$3

# index slurm array to grab subject
subject=${subjs[${SLURM_ARRAY_TASK_ID}]}

# Define scratch directory
scratch=/om2/scratch/tmp/$(whoami)/HBN_SUIT/$subject # assign working directory
export SUITENV_SUBJECTS_DIR=$scratch

# assign output directory
output_dir=${anat_dir}/suit_${version}

# Copy anatomicals to scratch directory
pushd $anat_dir
#t1_file=$subject/ses*/anat/*T1w.nii.gz
#t2_file=$subject/ses*/anat/*T2w.nii.gz
t1_file=$subject/ses*/anat/*desc_preproc_T1w.nii.gz
t2_file=$subject/ses*/anat/*desc_preproc_T2w.nii.gz
cp -nL $anat_dir/$t1_file $scratch/t1.nii.gz

# Account for presence of T2
if [ -e $t2_file ]; then
cp -nL $anat_dir/$t2_file $scratch/t2.nii.gz
t2_cmd_text="${scratch}/t2.nii.gz"
else t2_cmd_text=''
fi
popd

# Define the command
pushd $scratch
#cmd="singularity exec -e -B ${scratch},$scratch/license.txt:/usr/local/freesurfer/.license $IMG recon-all -subject $subject -i $scratch/t1.nii.gz $t2_cmd_text -all -qcache -hires -openmp 8"
cmd="matlab -nodisplay -r "$code_dir/run_suit(\"SUIT:run_normalization\", \"$scratch/t1.nii.gz\", \"$t2_cmd_text\", \"/om2/user/maedbh/bin/spm12\"); quit;""

# Run the command
echo "Submitted job for: ${subject}"
echo "$'Command :\n'${cmd}"
${cmd}

# copy files back
mkdir -p $output_dir
cp -nr $scratch/$subject/ $output_dir/
popd