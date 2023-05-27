#!/bin/bash
specs=($@)

### SET VARIABLES ###
model_name=anxiety
target=target_DX_01_Cat_new_binarize-spec.json
features=("features-all-all-all-all-spec.json" "features-Parent_Measures-all-all-all-spec.json" "features-Child_Measures-all-all-all-spec.json" "features-Teacher_Measures-all-all-all-spec.json")

### SET DIRECTORIES ###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE
out_dir=/om2/user/$(whoami)/hbn_data/interim/models/$model_name-models # OUTPUT DIR

mkdir -p $out_dir

# Get spec names (binary only) from the directory
if [[ $# -eq 0 ]]; then
    pushd $base/model_specs/participant_specs
    specs=($(ls *$model_name* | egrep -v multilabel | egrep -v adhd-age))
    popd
fi

# take the length of the array
# this will be useful for indexing later
len=$(expr ${#specs[@]} - 1) 

echo Spawning ${#specs[@]} spec-jobs.

# loop over features
for feature in "${features[@]}"
do
    sbatch --array=0-$len $bash_scripts/run_phenotypic_models.sh $base ${specs[@]} $out_dir $target $feature
done;