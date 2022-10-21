#!/bin/bash
######## Job Name: Test_Job ########
#SBATCH -J Generate_dice_csv_sep
#SBATCH -o logs/Generate_dice_csv_sep.o%j
#SBATCH -e logs/Generate_dice_csv_sep.e%j
######## Number of nodes: 1 ########
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 10
#SBATCH --gres gpu:tesla_v100:1
#SBATCH --mem 64G 
#SBATCH -t 05:00:00

cd /home/tang.zitian/nnUNet

######## Load module environment required for the job ########
module load cuda/11.3
source activate base

python nnunet/inference/dice_loss.py