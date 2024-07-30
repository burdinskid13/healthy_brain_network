#!/usr/bin/env bash
participants=($@)

# Model outname
model_outname=gender_mid_may/

### SET DIRECTORIES ###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
spec_dir=$base/model_specs # MODEL SPECS ARE STORED HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE
data_dir=/om2/user/$(whoami)/hbn_data/interim/phenotypes # INTERIM DATA ARE STORED HERE
model_dir=/om2/user/$(whoami)/hbn_data/interim/models/$model_outname # MODEL OUTPUT IS STORED HERE

### SET VARIABLES ###
pydraml=pydraml3-spec.json # pydraml base
target=target-Gender-spec.json # target spec

# features=("features-Child-language-spec.json" "features-Child-production-spec.json" "features-Child-intelligence-spec.json" "features-Child-reading-spec.json" "features-Child-emotional-status-spec.json" "features-Parent-Stress-spec.json" "features-all-internalizing-externalizing-spec.json")
participants=("participants-Reading-Only-spec.json" "participants-Reading-Only-early-spec.json" "participants-Reading-Only-emerging-spec.json" "participants-Reading-Only-fluent-spec.json")
features=("features-Child-internalizing-spec.json" "features-Child-externalizing-spec.json" "features-Parent-internalizing-spec.json" "features-Parent-externalizing-spec.json" "features-Teacher-internalizing-spec.json" "features-Teacher-externalizing-spec.json" "features-all-externalizing-spec.json" "features-all-internalizing-spec.json")

participants=("participants-Reading-Only-early-spec.json" "participants-Reading-Only-emerging-spec.json" "participants-Reading-Only-fluent-spec.json")
features=("features-Child-reading-spec.json")

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
