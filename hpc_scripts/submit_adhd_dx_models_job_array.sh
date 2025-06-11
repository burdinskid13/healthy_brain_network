#!/usr/bin/env bash
participants=($@)

# Model outname
model_outname=june_adhd_dx/

### SET DIRECTORIES ###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
spec_dir=$base/model_specs # MODEL SPECS ARE STORED HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE
data_dir=/om2/user/$(whoami)/hbn_data/interim/phenotypes # INTERIM DATA ARE STORED HERE
model_dir=/om2/user/$(whoami)/hbn_data/interim/models/$model_outname # MODEL OUTPUT IS STORED HERE

### SET VARIABLES ###
pydraml=pydraml3-spec.json # pydraml base
target=target-Diagnosis-ADHD-spec.json # target spec
# target=target-Diagnosis-ADHD-Subtype-spec.json # target spec

features=(
    "features-all-questions-spec.json" 
    )
 
participants=(
    "participants-adhd-all-male-spec.json" \
    # "participants-adhd-only-female-spec.json" \
    # "participants-adhd-only-prepubertal-spec.json" \
    # "participants-adhd-only-postpubertal-spec.json" \
    # "participants-adhd-only-female-prepubertal-spec.json" \
    # "participants-adhd-only-male-prepubertal-spec.json" \
    # "participants-adhd-only-female-postpubertal-spec.json" \
    # "participants-adhd-only-male-postpubertal-spec.json" \
    )

# participants=(
    # "participants-adhd-combined_type-male-spec.json" \
    # "participants-adhd-inattentive_type-female-spec.json" \
    # "participants-adhd-combined_type-female-spec.json" \
    # "participants-adhd-inattentive_type-male-spec.json" \
    # "participants-adhd-combined_type-prepubertal-spec.json" \
    # "participants-adhd-combined_type-postpubertal-spec.json" \
    # "participants-adhd-inattentive_type-prepubertal-spec.json" \
    # participants-adhd-inattentive_type-postpubertal-spec.json \
    # "participants-adhd-combined_type-male-prepubertal-spec.json" \
    # "participants-adhd-combined_type-male-postpubertal-spec.json" \
    # "participants-adhd-inattentive_type-male-prepubertal-spec.json" \
    # participants-adhd-inattentive_type-male-postpubertal-spec.json \
    # "participants-adhd-combined_type-female-prepubertal-spec.json" \
    # "participants-adhd-combined_type-female-postpubertal-spec.json" \
    # "participants-adhd-inattentive_type-female-prepubertal-spec.json" \
    # participants-adhd-inattentive_type-female-postpubertal-spec.json \
    # )

mkdir -p $model_dir

# take the length of the array - this will be useful for indexing later
len=$(expr ${#participants[@]} - 1) 

echo Spawning ${#participants[@]} spec-jobs.

# loop over features
for feature in "${features[@]}"
do
    sbatch --array=0-$len $bash_scripts/train_models.sh $base ${participants[@]} $model_dir $data_dir $spec_dir $target $feature $pydraml
done;
