"""
Methods related to dice calculation/generation
"""

# Imports

import numpy as np
import os
import nibabel as nib
from medpy import metric
import csv
from nnunet.evaluation.region_based_evaluation import create_region_from_mask

# Methods

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

def evaluate_case(image_pred, image_gt, regions, name):
    results = []
    # results.append(name[:15])
    # results.append(name.split('.')[0][-2:])
    for k, r in regions.items():
        mask_pred = create_region_from_mask(image_pred, r)
        mask_gt = create_region_from_mask(image_gt, r)
        
        dc = np.nan if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0 else metric.dc(mask_pred, mask_gt)
        results.append(dc)
    results.append(dice(image_pred, image_gt)) 
    return results # Dice of WT, TC, ET, NET/NCR, ALL classes

def generate_dice(pred_path, gt_path, curr_filename):
    image_pred = nib.load(os.path.join(pred_path, curr_filename)).get_fdata().astype(np.uint8)
    image_gt = nib.load(os.path.join(gt_path, curr_filename[:15] + "gt.nii.gz")).get_fdata().astype(np.uint8)
    dice = evaluate_case(image_pred, image_gt, get_brats_regions(), curr_filename)
    return dice

###############################
# modify this section only
modals = "NT1"
###############################


# path = "/scratch/tang.zitian/nnUNet_predictions/nnunet_sep400_predictions"
path = f"/scratch/tang.zitian/nnUNet_predictions/sep_{modals}_predictions"
gt_path = "/scratch/tang.zitian/nnUNet_raw_data_base/nnUNet_raw_data/Task400_BraTS2021/labelsTs"
csv_folder = "/home/tang.zitian/nnUNet/dice_loss_csv"

currdice = []

count = 0
for f in os.listdir(path):
    if "combined.nii.gz" in f:
        dice0 = generate_dice(path, gt_path, f)
        currdice.append(dice0)
        
        count += 1
        print("currently computing %d..." % count)

print("saving lists to csv files... :)")

with open(os.path.join(csv_folder, f"sep_{modals}_voted.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerows(currdice)