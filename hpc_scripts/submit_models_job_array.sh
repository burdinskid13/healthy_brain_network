#!/bin/bash
specs=($@)

### SET VARIABLES ###
participants=reading-age
model=pydraml_spec5.json
features_dir=remove_total_scores_demographics
target=$features_dir/target_DX_01_Cat_new_binarize-spec.json
features=("$features_dir/features-all-all-all-all-spec.json" "$features_dir/features-Parent_Measures-all-all-all-spec.json" "$features_dir/features-Child_Measures-all-all-all-spec.json" "$features_dir/features-Teacher_Measures-all-all-all-spec.json")
outname=$participants-$features_dir-bestclassifier-models

### SET DIRECTORIES ###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE
out_dir=/om2/user/$(whoami)/hbn_data/interim/models/$outname # OUTPUT DIR

mkdir -p $out_dir

# # Get spec names (binary only) from the directory
# if [[ $# -eq 0 ]]; then
#     pushd $base/model_specs/participant_specs
#     specs=($(ls *$participants* | egrep -v multilabel | egrep -v reading-age))
#     popd
# fi

specs=(spec-reading-age-male-05-06.json spec-reading-age-male-07-08.json spec-reading-age-male-09-10.json spec-reading-age-male-11-12.json spec-reading-age-male-13-14.json spec-reading-age-male-15-16.json spec-reading-age-male-17+.json spec-reading-age-male-05.json spec-reading-age-male-06.json spec-reading-age-male-07.json spec-reading-age-male-08.json spec-reading-age-male-09.json spec-reading-age-male-10.json spec-reading-age-male-11.json spec-reading-age-male-12.json spec-reading-age-male-13.json spec-reading-age-male-14.json spec-reading-age-male-15.json spec-reading-age-male-16.json spec-reading-age-female-05-06.json spec-reading-age-female-07-08.json spec-reading-age-female-09-10.json spec-reading-age-female-11-12.json spec-reading-age-female-13-14.json spec-reading-age-female-15-16.json spec-reading-age-female-17+.json spec-reading-age-female-05.json spec-reading-age-female-06.json spec-reading-age-female-07.json spec-reading-age-female-08.json spec-reading-age-female-09.json spec-reading-age-female-10.json spec-reading-age-female-11.json spec-reading-age-female-12.json spec-reading-age-female-13.json spec-reading-age-female-14.json spec-reading-age-female-15.json spec-reading-age-female-16.json)

# take the length of the array
# this will be useful for indexing later
len=$(expr ${#specs[@]} - 1) 

echo Spawning ${#specs[@]} spec-jobs.

# loop over features
for feature in "${features[@]}"
do
    sbatch --array=0-$len $bash_scripts/run_phenotypic_models.sh $base ${specs[@]} $out_dir $target $feature $model
done;