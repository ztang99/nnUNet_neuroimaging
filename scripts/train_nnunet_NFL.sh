#!/bin/bash
######## Job Name: Test_Job ########
#SBATCH -J Train_NFL
#SBATCH -o logs/Train_NFL.o%j
#SBATCH -e logs/Train_NFL.e%j
######## Number of nodes: 1 ########
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 10
#SBATCH --gres gpu:tesla_v100S:1
#SBATCH --mem 64G 
#SBATCH -t 48:00:00
 
cd /home/tang.zitian/nnUNet
# run script: sbatch scripts/train_nnunet_NFL.sh

######## Load module environment required for the job ########
module load cuda/11.3
source activate base

nnunet_use_progress_bar=1 nnUNet_n_proc_DA=10 nnUNet_train 3d_fullres nnUNetTrainer_NFL Task200_BraTS2021 1 --npz -p nnUNetPlansv2.1_verybig

