#!/bin/bash
#SBATCH --job-name=h200_hold_for_attach
#SBATCH --partition=h200_normal_q
#SBATCH --qos=tc_h200_normal_base
#SBATCH --account=gpuperfportability
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=afarhana@vt.edu

echo "started jobid $SLURM_JOB_ID on $(hostname) at $(date)"
sleep 5h
