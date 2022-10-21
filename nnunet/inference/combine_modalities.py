"""
combine the four modalities (4 nii.gz files generated during inference)
"""

import numpy as np
import nibabel as nib
import os

sep_path = "/scratch/tang.zitian/nnUNet_predictions/nnunet_sep400_predictions"
gt_path = "/scratch/tang.zitian/nnUNet_raw_data_base/nnUNet_raw_data/Task400_BraTS2021/labelsTs"

# t1 = nib.load(os.path.join(sep_path, "BraTS2021_01151T1.nii.gz")).get_fdata().astype(np.uint8)
# t1c = nib.load(os.path.join(sep_path, "BraTS2021_01151T1c.nii.gz")).get_fdata().astype(np.uint8)
# t2 = nib.load(os.path.join(sep_path, "BraTS2021_01151T2.nii.gz")).get_fdata().astype(np.uint8)
# fl = nib.load(os.path.join(sep_path, "BraTS2021_01151Flair.nii.gz")).get_fdata().astype(np.uint8)

def stack_modals(t1, t1c, t2, fl):
    allmodals = np.stack((t1, t1c, t2, fl), axis = 0)
    return allmodals

def return_all_modals(sep_path, gt_path, curr_filename, type):
    
    t1 = nib.load(os.path.join(sep_path, curr_filename + "T1.nii.gz")).get_fdata().astype(np.uint8)
    t1_gt = nib.load(os.path.join(gt_path, curr_filename + "T1.nii.gz")).get_fdata().astype(np.uint8)
    t1c = nib.load(os.path.join(sep_path, curr_filename + "T1c.nii.gz")).get_fdata().astype(np.uint8)
    t1c_gt = nib.load(os.path.join(sep_path, curr_filename + "T1c.nii.gz")).get_fdata().astype(np.uint8)
    t2 = nib.load(os.path.join(sep_path, curr_filename + "T2.nii.gz")).get_fdata().astype(np.uint8)
    t2_gt = nib.load(os.path.join(sep_path, curr_filename + "T2.nii.gz")).get_fdata().astype(np.uint8)
    fl = nib.load(os.path.join(sep_path, curr_filename + "Flair.nii.gz")).get_fdata().astype(np.uint8)
    fl_gt = nib.load(os.path.join(sep_path, curr_filename + "Flair.nii.gz")).get_fdata().astype(np.uint8)
    if type == "image":
        return stack_modals(t1, t1c, t2, fl)
    elif type == "gt":
        return stack_modals(t1_gt, t1c_gt, t2_gt, fl_gt)
    else:
        return "Type error: type must be 'image' or 'gt'."
    

