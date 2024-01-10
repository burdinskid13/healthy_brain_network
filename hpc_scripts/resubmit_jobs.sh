#!/bin/bash

# Get the job ID of the failed job
job_id=$(scontrol show job -n failed | grep $SLURM_JOB_ID | awk '{print $1}')

# Check if the job has already been resubmitted
if [[ $job_id != "" ]]; then
  # Check if the job is still in the PENDING state
  if [[ $(scontrol show job $job_id | grep State | awk '{print $2}') == "PENDING" ]]; then
    echo "Job $job_id is already being resubmitted. Skipping."
    exit 0
  fi
fi

# Resubmit the job
sbatch $SLURM_SUBMIT_FILE



### TO RUN THIS COMMAND:sbatch --requeue --signal=USR1