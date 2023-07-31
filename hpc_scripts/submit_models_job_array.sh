#!/bin/bash
specs=($@)

### SET VARIABLES ###
participants=adhd
model=pydraml_spec3.json
features_dir=basic_demographics
target=$features_dir/target_DX_01_Cat_new_binarize-spec.json
features=("$features_dir/features-all-all-all-all-spec.json" "$features_dir/features-Parent_Measures-all-all-all-spec.json" "$features_dir/features-Child_Measures-all-all-all-spec.json" "$features_dir/features-Teacher_Measures-all-all-all-spec.json")
outname=$participants-$features_dir-multiple-classifiers-models

### SET DIRECTORIES ###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE
out_dir=/om2/user/$(whoami)/hbn_data/interim/models/$outname # OUTPUT DIR

mkdir -p $out_dir

# Get spec names (binary only) from the directory
if [[ $# -eq 0 ]]; then
    pushd $base/model_specs/participant_specs
    specs=($(ls *$participants* | egrep -v multilabel | egrep -v adhd-age))
    popd
fi


#specs=(spec-adhd-age-05_06.json spec-adhd-age-07_08.json spec-adhd-age-09_10.json spec-adhd-age-11_12.json spec-adhd-age-13_14.json spec-adhd-age-15_16.json spec-adhd-age-17+.json spec-adhd-age-male-05_06.json spec-adhd-age-male-07_08.json spec-adhd-age-male-09_10.json spec-adhd-age-male-11_12.json spec-adhd-age-male-13_14.json spec-adhd-age-male-15_16.json spec-adhd-age-male-17+.json spec-adhd-age-female-05_06.json spec-adhd-age-female-07_08.json spec-adhd-age-female-09_10.json spec-adhd-age-female-11_12.json spec-adhd-age-female-13_14.json spec-adhd-age-female-15_16.json spec-adhd-age-female-17+.json)

# take the length of the array
# this will be useful for indexing later
len=$(expr ${#specs[@]} - 1) 

echo Spawning ${#specs[@]} spec-jobs.

# loop over features
for feature in "${features[@]}"
do
    sbatch --array=0-$len $bash_scripts/run_phenotypic_models.sh $base ${specs[@]} $out_dir $target $feature $model
done;