#!/bin/bash
######## Job Name: Test_Job ########
#SBATCH -J Train_T1cT2
#SBATCH -o logs/Train_T1cT2.o%j
#SBATCH -e logs/Train_T1cT2.e%j
######## Number of nodes: 1 ########
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 10
#SBATCH --gres gpu:tesla_a100:1
#SBATCH --mem 64G 
#SBATCH -t 48:00:00

cd /home/tang.zitian/nnUNet
# run script: sbatch scripts/train_nnunet_T1cT2.sh

######## Load module environment required for the job ########
module load cuda/11.3
source activate base

nnunet_use_progress_bar=1 nnUNet_n_proc_DA=10 nnUNet_train 3d_fullres nnUNetTrainer_T1cT2 Task200_BraTS2021 1 --npz -p nnUNetPlansv2.1_verybig

