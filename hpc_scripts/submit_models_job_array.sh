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
pydraml=pydraml2-spec.json # pydraml base
target=target-Diagnosis-spec.json # target spec
# participants=("participants-Reading-all-spec.json" "participants-Reading-female-spec.json" "participants-Reading-male-spec.json" "participants-Reading-early-readers-spec.json" "participants-Reading-fluent-readers-spec.json" "participants-Reading-emerging-readers-spec.json" "participants-Reading-early-readers-female-spec.json" "participants-Reading-fluent-readers-female-spec.json" "participants-Reading-emerging-readers-female-spec.json" "participants-Reading-early-readers-male-spec.json" "participants-Reading-fluent-readers-male-spec.json" "participants-Reading-emerging-readers-male-spec.json" "participants-Reading-Black-spec.json" "participants-Reading-White-spec.json") # participants
# features=("features-all-externalizing-spec.json" "features-all-internalizing-spec.json" "features-Child-language-spec.json" "features-Child-phonological-spec.json" "features-Child-production-spec.json" "features-Child-executive-function-spec.json" "features-Child-intelligence-spec.json" "features-Child-achievement-spec.json" "features-Child-emotional-status-spec.json" "features-Child-remove-total-scores-spec.json" "features-Parent-remove-total-scores-spec.json" "features-Teacher-remove-total-scores-spec.json" "features-all-demos-spec.json" "features-parent-child-mind-institute-spec.json" "features-child-cbcl-spec.json")

features=("features-Child-remove-total-scores-spec.json" "features-Parent-remove-total-scores-spec.json" "features-Teacher-remove-total-scores-spec.json")
participants=("participants-Reading-all-spec.json" "participants-Reading-female-spec.json" "participants-Reading-male-spec.json")

mkdir -p $model_dir

# take the length of the array - this will be useful for indexing later
len=$(expr ${#participants[@]} - 1) 

echo Spawning ${#participants[@]} spec-jobs.

# make specs
python3 $base/hbn/scripts/make_specs.py --out_dir=$$base/model_specs

# loop over features
for feature in "${features[@]}"
do
    sbatch --array=0-$len $bash_scripts/run_models.sh $base ${participants[@]} $model_dir $data_dir $spec_dir $target $feature $pydraml
done;

# get overall summary of models
# sbatch $bash_scripts/summarize_models.sh $base $model_dir