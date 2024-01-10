#!/bin/bash
specs=($@)

### SET VARIABLES ###
participants=reading
features_dir=basic_demographics
model_dir=multiple-classifiers
model_name=$participants-$features_dir-$model_dir-models
outname=$participants-$features_dir-secondlevel-$model_dir-models

### SET DIRECTORIES ###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE
model_dir=/om2/user/$(whoami)/hbn_data/interim/models/$model_name # MODEL DIR
out_dir=/om2/user/$(whoami)/hbn_data/interim/models/$outname # OUTPUT DIR

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