#!/usr/bin/env bash
participants=($@)

# Model outname
model_outname=reading_march/

### SET DIRECTORIES ###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
spec_dir=$base/model_specs # MODEL SPECS ARE STORED HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE
data_dir=/om2/user/$(whoami)/hbn_data/interim/phenotypes # INTERIM DATA ARE STORED HERE
model_dir=/om2/user/$(whoami)/hbn_data/interim/models/$model_outname # MODEL OUTPUT IS STORED HERE

### SET VARIABLES ###
pydraml=pydraml3-spec.json # pydraml base
target=target-Diagnosis-spec.json # target spec

features=("features-Teacher-remove-total-scores-spec.json")
# participants=("participants-Reading-all-spec.json" "participants-Reading-male-spec.json" "participants-Reading-female-spec.json" "participants-Reading-early-readers-spec.json" "participants-Reading-fluent-readers-spec.json" "participants-Reading-emerging-readers-spec.json" "participants-Reading-early-readers-female-spec.json" "participants-Reading-fluent-readers-female-spec.json" "participants-Reading-emerging-readers-female-spec.json" "participants-Reading-early-readers-male-spec.json" "participants-Reading-fluent-readers-male-spec.json" "participants-Reading-emerging-readers-male-spec.json" "participants-Reading-Black-spec.json" "participants-Reading-White-spec.json")
participants=("participants-Reading-female-spec.json")


mkdir -p $model_dir

# take the length of the array - this will be useful for indexing later
len=$(expr ${#participants[@]} - 1) 

echo Spawning ${#participants[@]} spec-jobs.

# make specs
sbatch $bash_scripts/make_specs.sh

# loop over features
for feature in "${features[@]}"
do
    sbatch --array=0-$len $bash_scripts/train_models.sh $base ${participants[@]} $model_dir $data_dir $spec_dir $target $feature $pydraml
done;

# get overall summary of models
# sbatch $bash_scripts/summarize_models.sh $base $model_dir