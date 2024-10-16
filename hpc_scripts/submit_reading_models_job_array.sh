#!/usr/bin/env bash
participants=($@)

# Model outname
model_outname=reading_augustC/

### SET DIRECTORIES ###
base=/om2/user/$(whoami)/healthy_brain_network # PUT YOUR REPO HERE
spec_dir=$base/model_specs # MODEL SPECS ARE STORED HERE
bash_scripts=$base/hpc_scripts/ # BASH SCRIPTS ARE HERE
data_dir=/om2/user/$(whoami)/hbn_data/interim/phenotypes # INTERIM DATA ARE STORED HERE
model_dir=/om2/user/$(whoami)/hbn_data/interim/models/$model_outname # MODEL OUTPUT IS STORED HERE

### SET VARIABLES ###
pydraml=pydraml3-spec.json # pydraml base
target=target-Diagnosis-Reading-spec.json # target spec

features=(
        # "features-Child-language-spec.json" \
        "features-Child-phonological-minimal-spec.json" \
        # "features-Child-production-spec.json" \
        # "features-Child-executive-function-spec.json" \
        # "features-Child-intelligence-spec.json" \
        # "features-Child-emotional-status-spec.json" \
        # "features-Parent-SES-spec.json" \
        # "features-Parent-Stress-spec.json" \
        # "features-Parent-Psychological-Function-spec.json" \
        # "features-Parent-Intake-Interview-spec.json" \
        # "features-Parent-Family-History-spec.json" \
        # "features-all-internalizing-spec.json" \
        # "features-all-externalizing-spec.json" \
        # "features-child-internalizing-spec.json" \
        # "features-parent-internalizing-spec.json" \
        # "features-parent-externalizing-spec.json" \
        # "features-all-demos-spec.json" \
        # "features-Child-reading-all-spec.json" \
        # "features-Child-reading-spec.json" \
        # "features-Child-reading-minimal-spec.json" \
        # "features-Child-reading-TOWRE-spec.json" \
        # "features-Child-reading-WIAT-spec.json" \
        )

participants=(
    # 'participants-reading-only-Black-spec.json' \
    # 'participants-reading-only-White-spec.json' \
    # 'participants-reading-all-Black-spec.json' \
    # 'participants-reading-all-White-spec.json' \
    # 'participants-no-reading-spec.json' \
    # 'participants-reading-all-early-readers-spec.json' \
    # 'participants-reading-all-emerging-readers-spec.json' \
    # 'participants-reading-all-fluent-readers-spec.json' \
    # 'participants-reading-all-female-spec.json' \
    'participants-reading-all-male-spec.json' \
    'participants-reading-all-spec.json' \
    'participants-reading-only-spec.json' \
)

# participants=(
#              'participants-reading-all-early-readers-spec.json' \
#              'participants-reading-all-emerging-readers-spec.json' \
#              'participants-reading-all-fluent-readers-spec.json' \
#             )

mkdir -p $model_dir

# take the length of the array - this will be useful for indexing later
len=$(expr ${#participants[@]} - 1) 

echo Spawning ${#participants[@]} spec-jobs.

# loop over features
for feature in "${features[@]}"
do
    sbatch --array=0-$len $bash_scripts/train_models.sh $base ${participants[@]} $model_dir $data_dir $spec_dir $target $feature $pydraml
done;
