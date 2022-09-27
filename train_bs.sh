#!/bin/bash
######## Job Name: Test_Job ########
#SBATCH -J Train_SEG_Job
#SBATCH -o logs/Train_SEG_Job.o%j
#SBATCH -e logs/Train_SEG_Job.e%j
######## Number of nodes: 1 ########
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 20
#SBATCH --gres gpu:tesla_v100:1
#SBATCH --mem 64G 
#SBATCH -t 48:00:00
 
cd /home/tang.zitian/nnUNet
​
######## Load module environment required for the job ########
module load cuda/11.3
source activate base # change to base if not using virtualenv
​
nnunet_use_progress_bar=1 nnUNet_n_proc_DA=20 nnUNet_train 3d_fullres nnUNetTrainerV2 Task101_BraTS2021 1 --npz -p nnUNetPlansv2.1_verybig 
# nnunet_use_progress_bar=1 nnUNet_n_proc_DA=20 nnUNet_train 2d nnUNetTrainerV2 Task100_ASME2022 0 --npz
​
# nnunet_use_progress_bar=1 nnUNet_n_proc_DA=20 nnUNet_train 3d_lowres nnUNetTrainerV2 Task100_ASME2022 0 --npz -p nnUNetPlansv2.1_verybig
​
# nnunet_use_progress_bar=1 nnUNet_n_proc_DA=20 nnUNet_train 3d_cascade_fullres nnUNetTrainerV2CascadeFullRes Task100_ASME2022 0 --npz -p nnUNetPlansv2.1_verybig