"""
Methods related to dice calculation/generation
"""

import numpy as np
import os
import nibabel as nib
from medpy import metric
from nnunet.evaluation.region_based_evaluation import create_region_from_mask
from nnunet.inference.combine_modalities import stack_modals, return_all_modals


def dice(prediction, target):
    prediction = prediction.flatten()
    target = target.flatten()
    TP = np.sum(prediction[prediction != 0] == target[prediction != 0])
    TN = np.sum(target[prediction == 0] == 0)
    FP = np.sum(target[prediction != 0] == 0)
    FN = np.sum(target[prediction == 0] != 0)

    sensitivity = TP / (TP + FN)
    specifity = TN / (TN + FP)
    dice = (2 * TP) / ((FP + TP) + (TP + FN))

    return dice, sensitivity, specifity

def get_brats_regions():
    """
    this is only valid for the brats data in here where the labels are 1, 2, and 3. The original brats data have a
    different labeling convention!
    :return:
    """
    regions = {
        "whole tumor": (1, 2, 3),
        "tumor core": (2, 3),
        "enhancing tumor": (3,),
        "ncr/net": (2,),
        "ed": (1,),
    }
    return regions

# region = get_brats_regions()
# print(region)

def evaluate_case(image_pred, image_gt, regions):
    results = []
    for k, r in regions.items():
        mask_pred = create_region_from_mask(image_pred, r)
        mask_gt = create_region_from_mask(image_gt, r)
        
        dc = np.nan if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0 else metric.dc(mask_pred, mask_gt)
        results.append(dc)
    results.append(dice(image_pred, image_gt)) 
    return results # Dice of WT, TC, ET, NET/NCR, ALL classes

# allmodals_path = "/scratch/tang.zitian/nnUNet_predictions/nnunet_allmodals_predictions"
# pred_path contains nii.gz, npz, and pkl files, plus plans.pkl and .txt (941 total)

# NT1path = "/scratch/tang.zitian/nnUNet_predictions/nnunet_NT1_predictions"
# NT1cpath = "/scratch/tang.zitian/nnUNet_predictions/nnunet_NT1c_predictions"
# NT2path = "/scratch/tang.zitian/nnUNet_predictions/nnunet_NT2_predictions"
# NFLpath = "/scratch/tang.zitian/nnUNet_predictions/nnunet_NFL_predictions"
# T1path = "/scratch/tang.zitian/nnUNet_predictions/nnunet_T1_predictions"
# T1cpath = "/scratch/tang.zitian/nnUNet_predictions/nnunet_T1c_predictions"
# T2path = "/scratch/tang.zitian/nnUNet_predictions/nnunet_T2_predictions"
# FLpath = "/scratch/tang.zitian/nnUNet_predictions/nnunet_FL_predictions"
# T1cT2path = "/scratch/tang.zitian/nnUNet_predictions/nnunet_T1cT2_predictions"
# T1cFLpath = "/scratch/tang.zitian/nnUNet_predictions/nnunet_T1cFL_predictions"
# T1T2path = "/scratch/tang.zitian/nnUNet_predictions/nnunet_T1T2_predictions"
# T1FLpath = "/scratch/tang.zitian/nnUNet_predictions/nnunet_T1FL_predictions"
# T1T1cpath = "/scratch/tang.zitian/nnUNet_predictions/nnunet_T1T1c_predictions"
# T2FLpath = "/scratch/tang.zitian/nnUNet_predictions/nnunet_T2FL_predictions"

sep_path = "/scratch/tang.zitian/nnUNet_predictions/nnunet_sep400_predictions"
gt_path = "/scratch/tang.zitian/nnUNet_raw_data_base/nnUNet_raw_data/Task400_BraTS2021/labelsTs"

# gt_path = "/scratch/tang.zitian/nnUNet_raw_data_base/nnUNet_raw_data/Task200_BraTS2021/labelsTs"
# gt_path contains only nii.gz files (total 313)
# current_modals = allmodals_path.split('/')[-1].split('_')[-2] #allmodals/NT1/etc.

def generate_dice(pred_path, gt_path, curr_filename):
    image_pred = nib.load(os.path.join(pred_path, curr_filename)).get_fdata().astype(np.uint8)
    image_gt = nib.load(os.path.join(gt_path, curr_filename)).get_fdata().astype(np.uint8)
    dice = evaluate_case(image_pred, image_gt, get_brats_regions())
    return dice

def generate_dice_sep(pred_path, gt_path, curr_filename):
    image_pred = return_all_modals(pred_path, gt_path, curr_filename[:15], "image")
    image_gt = return_all_modals(pred_path, gt_path, curr_filename[:15], "gt")
    dice = evaluate_case(image_pred, image_gt, get_brats_regions())
    return dice

# allmodals_dice = []
# NT1_dice = []
# NT1c_dice = []
# NT2_dice = []
# NFL_dice = []
# T1_dice = []
# T1c_dice = []
# T2_dice = []
# FL_dice = []
# T1T1c_dice = []
# T1T2_dice = []
# T1FL_dice = []
# T1cT2_dice = []
# T1cFL_dice = []
# T2FL_dice = []
sep_dice = []

t1 = []
t1c = []
t2 = []
fl = []

for f in os.listdir(sep_path):
    if "nii.gz" in f:
        if "T1c" in f:
            t1c.append(f)
        elif "T1" in f:
            t1.append(f)
        elif "T2" in f:
            t2.append(f)
        elif "Flair" in f:
            fl.append(f)

print(len(t1c))
print(t1c)

for i in t1:
    dice0 = generate_dice_sep(sep_path, gt_path, f)
    sep_dice.append(dice0)

# for f in os.listdir(gt_path):
#     dice1 = generate_dice(allmodals_path, gt_path, f)
#     allmodals_dice.append(dice1)
#     dice2 = generate_dice(NT1path, gt_path, f)
#     NT1_dice.append(dice2)
#     dice3 = generate_dice(NT1cpath, gt_path, f)
#     NT1c_dice.append(dice3)
#     dice4 = generate_dice(NT2path, gt_path, f)
#     NT2_dice.append(dice4)
#     dice5 = generate_dice(NFLpath, gt_path, f)
#     NFL_dice.append(dice5)
#     dice6 = generate_dice(T1path, gt_path, f)
#     T1_dice.append(dice6)
#     dice7 = generate_dice(T1cpath, gt_path, f)
#     T1c_dice.append(dice7)
#     dice8 = generate_dice(T2path, gt_path, f)
#     T2_dice.append(dice8)
#     dice9 = generate_dice(FLpath, gt_path, f)
#     FL_dice.append(dice9)
#     dice10 = generate_dice(T1T1cpath, gt_path, f)
#     T1T1c_dice.append(dice10)
#     dice11 = generate_dice(T1T2path, gt_path, f)
#     T1T2_dice.append(dice11)
#     dice12 = generate_dice(T1FLpath, gt_path, f)
#     T1FL_dice.append(dice12)
#     dice13 = generate_dice(T1cT2path, gt_path, f)
#     T1cT2_dice.append(dice13)
#     dice14 = generate_dice(T1cFLpath, gt_path, f)
#     T1cFL_dice.append(dice14)
#     dice15 = generate_dice(T2FLpath, gt_path, f)
#     T2FL_dice.append(dice15)

# print(allmodals_dice)
# print(len(allmodals_dice)) ###313

print("saving lists to csv files... :)")
import csv
# specify output folder
csv_folder = "/home/tang.zitian/nnUNet/dice_loss_csv"
with open(os.path.join(csv_folder,"sepmodals.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerows(sep_dice)

# with open(os.path.join(csv_folder,"allmodals.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(allmodals_dice)
# with open(os.path.join(csv_folder,"NT1.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(NT1_dice)
# with open(os.path.join(csv_folder,"NT1c.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(NT1c_dice)
# with open(os.path.join(csv_folder,"NT2.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(NT2_dice)
# with open(os.path.join(csv_folder,"NFL.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(NFL_dice)
# with open(os.path.join(csv_folder,"T1.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(T1_dice)
# with open(os.path.join(csv_folder,"T1c.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(T1c_dice)
# with open(os.path.join(csv_folder,"T2.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(T2_dice)
# with open(os.path.join(csv_folder,"FL.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(FL_dice)
# with open(os.path.join(csv_folder,"T1T1c.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(T1T1c_dice)
# with open(os.path.join(csv_folder,"T1T2.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(T1T2_dice)
# with open(os.path.join(csv_folder,"T1FL.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(T1FL_dice)
# with open(os.path.join(csv_folder,"T1cT2.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(T1cT2_dice)
# with open(os.path.join(csv_folder,"T1cFL.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(T1cFL_dice)
# with open(os.path.join(csv_folder,"T2FL.csv"), "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(T2FL_dice)


# image_pred = nib.load("/scratch/tang.zitian/nnUNet_predictions/nnunet_allmodals_predictions/BraTS2021_00016.nii.gz").get_fdata().astype(np.uint8)
# image_gt = nib.load("/scratch/tang.zitian/nnUNet_raw_data_base/nnUNet_raw_data/Task200_BraTS2021/labelsTs/BraTS2021_00016.nii.gz").get_fdata().astype(np.uint8)
# print(evaluate_case(image_pred, image_gt, get_brats_regions()))
