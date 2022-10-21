#!/bin/bash
######## Job Name: Test_Job ########
#SBATCH -J run_sep_modals
#SBATCH -o logs/run_sep_modals.o%j
#SBATCH -e logs/run_sep_modals.e%j
######## Number of nodes: 1 ########
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 10
#SBATCH --gres gpu:tesla_v100:1
#SBATCH --mem 64G 
#SBATCH -t 00:30:00

cd /home/tang.zitian/nnUNet/tang_testing
# run script: sbatch scripts/test_nnunet_T1.sh

######## Load module environment required for the job ########
module load cuda/11.3
source activate base

python separate_modals.py