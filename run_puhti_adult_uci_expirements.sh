#!/bin/bash -l
for i in {1..20}
do
   TASK_ID=$i sbatch puhti_adult_uci_experiment.sh
done