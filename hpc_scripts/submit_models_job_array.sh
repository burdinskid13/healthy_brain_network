#!/bin/bash
specs=($@)

### SET DIRECTORIES - YOU MAY HAVE TO CHANGE VIRTUAL ENVIRONMENT PATH###
source ~/.bash_profile # set paths
source ~/.bashrc # set paths
source /om2/user/$(whoami)/bin/miniconda3/bin/activate healthy-brain-network

# Model outname
model_outname=reading_winter_2023/reading_remove_total_scores_RandomForestClassifier

### SET DIRECTORIES ###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE
data_dir=/om2/user/$(whoami)/hbn_data/interim/phenotypes # INTERIM DATA ARE STORED HERE
model_dir=/om2/user/$(whoami)/hbn_data/interim/models/$model_outname # MODEL OUTPUT IS STORED HERE

### SET VARIABLES ###
pydraml=$base/model_specs/pydraml3-spec.json # pydraml base
target=$base/model_specs/target-Diagnosis-spec.json # target spec
features=("$base/model_specs/features-Child-remove-total-scores-spec.json" "$base/model_specs/features-Teacher-remove-total-scores-spec.json" "$base/model_specs/features-Parent-remove-total-scores-spec.json") # list of feature specs
specs=("$base/model_specs/participants-Reading-all-spec.json" "$base/model_specs/participants-Reading-male-spec.json" "$base/model_specs/participants-Reading-female-spec.json" "$base/model_specs/participants-Reading-White-spec.json" "$base/model_specs/participants-Reading-Black-spec.json" "$base/model_specs/participants-Reading-multiple-races-spec.json" "$base/model_specs/participants-Reading-Hispanic-spec.json") # participant specs

mkdir -p $model_dir

# # Get spec names (binary only) from the directory
# if [[ $# -eq 0 ]]; then
#     pushd $base/model_specs/participant_specs
#     specs=($(ls *$participants* | egrep -v multilabel | egrep -v reading-age))
#     popd
# fi

# take the length of the array - this will be useful for indexing later
len=$(expr ${#specs[@]} - 1) 

echo Spawning ${#specs[@]} spec-jobs.

# loop over features
for feature in "${features[@]}"
do
    sbatch --array=0-$len $bash_scripts/run_firstlevel_models.sh $base ${specs[@]} $model_dir $data_dir $target $feature $pydraml
done;

# get overall summary of models
sbatch $bash_scripts/summarize_models.sh $base $model_dir