#!/usr/bin/env bash
participants=($@)

# Model outname
model_outname=adhd_october/

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
#         "features-teacher-internalizing-spec.json" \
#         "features-teacher-externalizing-spec.json" \
#         "features-teacher-anxious_depressed-spec.json" \
#         "features-teacher-social_problems-spec.json" \
#         "features-teacher-thought_problems-spec.json" \
#         "features-teacher-withdrawn_depressed-spec.json" \
#         "features-teacher-attention_problems-spec.json" \
#         "features-teacher-rule_breaking-spec.json" \
#         "features-teacher-aggressive_behavior-spec.json" \
#         "features-teacher-somatic_complaints-spec.json" \
#         "features-child-internalizing-spec.json" \
#         "features-child-externalizing-spec.json" \
#         "features-child-anxious_depressed-spec.json" \
#         "features-child-withdrawn_depressed-spec.json" \
#         "features-child-social_problems-spec.json" \
#         "features-child-thought_problems-spec.json" \
#         "features-child-attention_problems-spec.json" \
#         "features-child-rule_breaking-spec.json" \
#         "features-child-aggressive_behavior-spec.json" \
#         "features-child-somatic_complaints-spec.json" \
#         "features-parent-withdrawn_depressed-spec.json" \
#         "features-parent-anxious_depressed-spec.json" \
#         "features-parent-externalizing-spec.json" \
#         "features-parent-internalizing-spec.json" \
#         "features-parent-thought_problems-spec.json" \
#         "features-parent-social_problems-spec.json" \
#         "features-parent-rule_breaking-spec.json" \
#         "features-parent-attention_problems-spec.json" \
#         "features-parent-somatic_complaints-spec.json" \
#         "features-parent-aggressive_behavior-spec.json" \
#         "features-demos-excl-sex-spec.json"
#         "features-child-defiance_aggression-spec.json" \
#         "features-child-family_relations-spec.json" \
#         "features-child-hyperactive_impulsivity-spec.json" \
#         "features-child-inattention-spec.json" \
#         "features-child-learning_problems-spec.json" \
#         "features-child-negative_impression-spec.json" \
#         "features-child-positive_impression-spec.json" \
        "features-child-swan_inattention-spec.json" \
        # "features-child-swan_hyperactive-spec.json" \
        # "features-child-suicidality-spec.json" \
        )

participants=(
    # "participants-adhd-all-comorbidities-spec.json" \
    "participants-adhd-no-comorbidities-spec.json" \
    # "participants-other_diagnoses-spec.json" \
    # "participants-no_diagnosis_given-spec.json" \
    # "participants-adhd_combined-spec.json" \
    # "participants-adhd_inattentive-spec.json" \ 
    # "participants-adhd_hyperactive-spec.json" \
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
