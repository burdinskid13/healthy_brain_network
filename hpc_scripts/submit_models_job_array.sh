#!/usr/bin/env bash
specs=($@)

# Model outname
model_outname=reading_winter_2023/reading_feature_models

### SET DIRECTORIES ###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE
data_dir=/om2/user/$(whoami)/hbn_data/interim/phenotypes # INTERIM DATA ARE STORED HERE
model_dir=/om2/user/$(whoami)/hbn_data/interim/models/$model_outname # MODEL OUTPUT IS STORED HERE

### SET VARIABLES ###
pydraml=$base/model_specs/pydraml3-spec.json # pydraml base
target=$base/model_specs/target-Diagnosis-spec.json # target spec
features=("$base/model_specs/features-Child-externalizing-spec.json" "$base/model_specs/features-Parent-externalizing-spec.json" "$base/model_specs/features-Teacher-externalizing-spec.json" "$base/model_specs/features-Child-internalizing-spec.json" "$base/model_specs/features-Parent-internalizing-spec.json" "$base/model_specs/features-Teacher-internalizing-spec.json") # features
specs=("$base/model_specs/participants-Reading-all-spec.json" "$base/model_specs/participants-Reading-female-spec.json" "$base/model_specs/participants-Reading-male-spec.json" "$base/model_specs/participants-Reading-early-readers-spec.json" "$base/model_specs/participants-Reading-fluent-readers-spec.json" "$base/model_specs/participants-Reading-emerging-readers-spec.json" "$base/model_specs/participants-Reading-early-readers-female-spec.json" "$base/model_specs/participants-Reading-fluent-readers-female-spec.json" "$base/model_specs/participants-Reading-emerging-readers-female-spec.json" "$base/model_specs/participants-Reading-early-readers-male-spec.json" "$base/model_specs/participants-Reading-fluent-readers-male-spec.json" "$base/model_specs/participants-Reading-emerging-readers-male-spec.json") # participants

mkdir -p $model_dir

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