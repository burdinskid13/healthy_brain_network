#!/usr/bin/env bash
participants=($@)

# Model outname
model_outname=adhd_august/

### SET DIRECTORIES ###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
spec_dir=$base/model_specs # MODEL SPECS ARE STORED HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE
data_dir=/om2/user/$(whoami)/hbn_data/interim/phenotypes # INTERIM DATA ARE STORED HERE
model_dir=/om2/user/$(whoami)/hbn_data/interim/models/$model_outname # MODEL OUTPUT IS STORED HERE

### SET VARIABLES ###
pydraml=pydraml3-spec.json # pydraml base
target=target-Gender-spec.json # target spec

features=(
        # "features-teacher-internalizing-spec.json" \
        # "features-teacher-externalizing-spec.json" \
        # "features-child-internalizing-spec.json" \
        # "features-child-externalizing-spec.json" \
        # "features-parent-externalizing-spec.json" \
        "features-parent-internalizing-spec.json" \
        )

participants=(
    # participants-adhd-all-comorbidities-spec.json \
    # 'participants-adhd-all-comorbidities-stage1-spec.json' \
    # 'participants-adhd-all-comorbidities-stage2-spec.json' \
    # 'participants-adhd-all-comorbidities-stage3-spec.json' \
    # 'participants-adhd-all-comorbidities-stage4-spec.json' \
    # 'participants-adhd-all-comorbidities-stage5-spec.json' \
    # 'participants-adhd-all-comorbidities-black-spec.json' \
    # 'participants-adhd-all-comorbidities-white-spec.json' \
    'participants-adhd-all-comorbidities-prepuberty-white-spec.json' \
    'participants-adhd-all-comorbidities-puberty-white-spec.json' \
    'participants-adhd-all-comorbidities-postpuberty-white-spec.json' \
    'participants-adhd-all-comorbidities-prepuberty-black-spec.json' \
    'participants-adhd-all-comorbidities-puberty-black-spec.json' \
    'participants-adhd-all-comorbidities-postpuberty-black-spec.json' \
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
