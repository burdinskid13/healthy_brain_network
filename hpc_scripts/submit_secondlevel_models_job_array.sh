#!/bin/bash
specs=($@)

### SET VARIABLES ###
model_name=adhd
outname=adhd-secondlevel

### SET DIRECTORIES ###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE
model_dir=/om2/user/$(whoami)/hbn_data/interim/models/$model_name-models # MODEL DIR
out_dir=/om2/user/$(whoami)/hbn_data/interim/models/$outname-models # OUTPUT DIR

mkdir -p $out_dir

# get model directories as specs
if [[ $# -eq 0 ]]; then
    pushd $model_dir
    specs=($(ls * -d))
    popd
fi

# take the length of the array
# this will be useful for indexing later
len=$(expr ${#specs[@]} - 1) 

echo Spawning ${#specs[@]} spec-jobs.

# submit job
sbatch --array=0-$len $bash_scripts/run_secondlevel_phenotypic_models.sh $base ${specs[@]} $model_dir $out_dir