#!/usr/bin/env bash
participants=($@)

# Model outname
model_outname=reading_august_OLA/

### SET DIRECTORIES ###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
spec_dir=$base/model_specs # MODEL SPECS ARE STORED HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE
data_dir=/om2/user/$(whoami)/hbn_data/interim/phenotypes # INTERIM DATA ARE STORED HERE
model_dir=/om2/user/$(whoami)/hbn_data/interim/models/$model_outname # MODEL OUTPUT IS STORED HERE

### SET VARIABLES ###
pydraml=pydraml4-spec.json # pydraml base
target=target-Continuous-Reading-spec.json # target spec

features=(
        "features-Child-language-spec.json" \
        "features-Child-phonological-spec.json" \
        "features-Child-production-spec.json" \
        "features-Child-executive-function-spec.json" \
        "features-Child-intelligence-spec.json" \
        "features-Child-emotional-status-spec.json" \
        "features-Child-reading-TOWRE-spec.json" \
        "features-Child-reading-WIAT-spec.json" \
        )

participants=(
        "participants-reading-all-comorbidities-early-readers-spec.json" \
        "participants-reading-all-comorbidities-emerging-readers-spec.json" \
        "participants-reading-all-comorbidities-fluent-readers-spec.json" \
        "participants-reading-no-comorbidities-early-readers-spec.json" \
        "participants-reading-no-comorbidities-emerging-readers-spec.json" \
        "participants-reading-no-comorbidities-fluent-readers-spec.json" \
        )

mkdir -p $model_dir

# take the length of the array - this will be useful for indexing later
len=$(expr ${#participants[@]} - 1) 

echo Spawning ${#participants[@]} spec-jobs.

# loop over features
for feature in "${features[@]}"
do
    sbatch --array=0-$len $bash_scripts/train_models.sh $base ${participants[@]} $model_dir $data_dir $spec_dir $target $feature $pydraml
done;
