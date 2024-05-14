#!/usr/bin/env bash
participants=($@)

# Model outname
model_outname=adhd_april/

### SET DIRECTORIES ###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
spec_dir=$base/model_specs # MODEL SPECS ARE STORED HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE
data_dir=/om2/user/$(whoami)/hbn_data/interim/phenotypes # INTERIM DATA ARE STORED HERE
model_dir=/om2/user/$(whoami)/hbn_data/interim/models/$model_outname # MODEL OUTPUT IS STORED HERE

### SET VARIABLES ###
pydraml=pydraml3-spec.json # pydraml base
target=target-Diagnosis-spec.json # target spec

features=("features-child-cbcl-spec.json" "features-child-anxiety-spec.json" "features-child-mood-spec.json" "features-child-suicide-spec.json" "features-child-language-all-spec.json" "features-parent-cbcl-spec.json" "features-parent-anxiety-spec.json" "features-parent-strengths-weaknesses-adhd-spec.json" "features-all-internalizing-spec.json" "features-all-externalizing-spec.json" "features-all-demos-spec.json" "features-Teacher-remove-total-scores-spec.json" "features-Parent-remove-total-scores-spec.json" "features-Child-remove-total-scores-spec.json" "features-parent-child-mind-institute-spec.json")
participants=("participants-adhd-male-spec.json" "participants-adhd-female-spec.json" "participants-adhd-all-spec.json")
features=("features-parent-social-communication-spec.json" "features-parent-strengths-weaknesses-adhd-spec.json" "features-parent-strengths-weaknesses-adhd-all-spec.json" "features-teacher-cbcl-spec.json" "features-child-language-all-spec.json")


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
