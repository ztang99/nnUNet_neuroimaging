#!/bin/bash
######## Job Name: Test_Job ########
#SBATCH -J Test_NT2
#SBATCH -o logs/Test_NT2.o%j
#SBATCH -e logs/Test_NT2.e%j
######## Number of nodes: 1 ########
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 10
#SBATCH --gres gpu:tesla_v100:1
#SBATCH --mem 64G 
#SBATCH -t 00:30:00

# cd /home/tang.zitian/nnUNet
cd /scratch/tang.zitian/nnUNet_trained_models/nnUNet/3d_fullres/Task200_BraTS2021/nnUNetTrainer_NT2__nnUNetPlansv2.1
# run script: sbatch scripts/test_nnunet_NT2.sh

######## Load module environment required for the job ########
module load cuda/11.3
source activate base

### INPUT_FOLDER: where your testing data located ###
### which is: /scratch/tang.zitian/nnUNet_preprocessed/Task200_BraTS2021/nnUNetData_plans_v2.1_verybig_stage0 ###
### OUTPUT_FOLDER: /scratch/tang.zitian/nnUNet_predictions
nnUNet_predict -tr nnUNetTrainer_NT2 -i /scratch/tang.zitian/nnUNet_raw_data_base/nnUNet_raw_data/Task200_BraTS2021/imagesTs -o /scratch/tang.zitian/nnUNet_predictions/nnunet_NT2_predictions -t Task200_BraTS2021 -m 3d_fullres --save_npz  -f 1 -p nnUNetPlansv2.1_verybig