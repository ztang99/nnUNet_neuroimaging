#!/bin/bash
######## Job Name: Test_Job ########
#SBATCH -J run_sep_modals
#SBATCH -o logs/run_sep_modals.o%j
#SBATCH -e logs/run_sep_modals.e%j
######## Number of nodes: 1 ########
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 64G 
#SBATCH -t 02:00:00

cd /home/tang.zitian/nnUNet
# run script: sbatch scripts/test_nnunet_T1.sh

######## Load module environment required for the job ########
source activate base

# python nnunet/dataset_conversion/Task300_BraTS_2021_sepmodals.py
nnUNet_plan_and_preprocess -t 400 --planner3d ExperimentPlanner3D_v21_32GB -pl2d None --verify_dataset_integrity