#!/bin/bash
specs=($@)

### SET VARIABLES ###
model_name=adhd-multilabel
target=target_DX_01_Cat_new_factorize-spec.json

### SET DIRECTORIES - MAY NEED TO BE CHANGED DEPENDING ON YOUR PATHS###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
out_dir=/om2/user/$(whoami)/hbn_data/interim/models/$model_name-models # MODELS WILL BE SAVED HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE

mkdir -p $out_dir

# Get participant spec names from the directory
if [[ $# -eq 0 ]]; then
    pushd $base/model_specs/participant_specs
    specs=($(ls spec-$model_name*.json))
    popd
fi

# take the length of the array
# this will be useful for indexing later
len=$(expr ${#specs[@]} - 1) 

echo Spawning ${#specs[@]} spec-jobs.

sbatch --array=0-$len $bash_scripts/run_phenotypic_models.sh $base ${specs[@]} $out_dir $target