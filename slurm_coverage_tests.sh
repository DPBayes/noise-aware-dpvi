#!/bin/bash -l
project_key=123
username=123
#SBATCH --job-name=coverage_test
#SBATCH --output=./logs/output/coverage_test_%A_%a.txt
#SBATCH --error=./logs/error/coverage_test_%A_%a.txt
#SBATCH --account=$project_key
#SBATCH --partition=small
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --array={ARRAY_FROM}-{ARRAY_TO}

source ~/.bashrc

# TODO: generate a the conda environment in the following path:
export PATH="/projappl/$project_key/$username/na-dpvi/bin:$PATH"

OUTPUT_DIR="/scratch/$project_key/$username/results-coverage/{EXPERIMENT_NAME}"
mkdir -p $OUTPUT_DIR
OUTPUT_DIR="/scratch/$project_key/$username/results-coverage/{EXPERIMENT_NAME}/{EPSILON}"
mkdir -p $OUTPUT_DIR
OUTPUT_DIR="/scratch/$project_key/$username/results-coverage/{EXPERIMENT_NAME}/{EPSILON}/{METHOD}"
mkdir -p $OUTPUT_DIR

DEBUG_OUTPUT_DIR="/scratch/$project_key/$username/results-coverage-debug/{EXPERIMENT_NAME}"
mkdir -p $DEBUG_OUTPUT_DIR
DEBUG_OUTPUT_DIR="/scratch/$project_key/$username/results-coverage-debug/{EXPERIMENT_NAME}/{EPSILON}"
mkdir -p $DEBUG_OUTPUT_DIR
DEBUG_OUTPUT_DIR="/scratch/$project_key/$username/results-coverage-debug/{EXPERIMENT_NAME}/{EPSILON}/{METHOD}"
mkdir -p $DEBUG_OUTPUT_DIR

srun python3 coverage_experiment.py --task_id $SLURM_ARRAY_TASK_ID --debug_output_directory $DEBUG_OUTPUT_DIR --output_directory $OUTPUT_DIR --target_epsilon {EPSILON} {USE_MCMC} --experiment_name {EXPERIMENT_NAME}