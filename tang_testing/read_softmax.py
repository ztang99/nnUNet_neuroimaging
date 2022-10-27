"""
This file read in and combine modalities softmax, then save to nii.gz final profile
"""

import numpy as np
import nibabel as nib
import os
# import SimpleITK as sitk


from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax, save_segmentation_nifti

modals = "T1T2"
# get list of subject ids
all_ids = []
for f in os.listdir(f"/scratch/tang.zitian/nnUNet_predictions/sep_{modals}_predictions"):
    if ("nii.gz" in f) & (f[:15] not in all_ids):
        all_ids.append(f[:15])

subject_id = "BraTS2021_00016"

# for subject_id in all_ids:
    # select whichever you needed
softmax1 = np.load(f"/scratch/tang.zitian/nnUNet_predictions/sep_{modals}_predictions/{subject_id}T1.npz")['softmax']
softmax2 = np.load(f"/scratch/tang.zitian/nnUNet_predictions/sep_{modals}_predictions/{subject_id}T2.npz")['softmax']
# T1c_softmax = np.load(f"/scratch/tang.zitian/nnUNet_predictions/sep_{modals}_predictions/{subject_id}T1c.npz")['softmax']
# Flair_softmax = np.load(f"/scratch/tang.zitian/nnUNet_predictions/sep_{modals}_predictions/{subject_id}Flair.npz")['softmax']
# gt1 = nib.load("/scratch/tang.zitian/nnUNet_raw_data_base/nnUNet_cropped_data/Task400_BraTS2021/gt_segmentations/BraTS2021_00016T1.nii.gz").get_fdata().astype(np.uint8)
out_gt = f"/scratch/tang.zitian/nnUNet_raw_data_base/nnUNet_raw_data/Task400_BraTS2021/labelsTs/{subject_id}combined.nii.gz"
# print(T1_softmax.shape, T2_softmax.shape)

out_fname = f"/scratch/tang.zitian/nnUNet_predictions/sep_{modals}_predictions/{subject_id}combined.nii.gz"

# compute combined & averaged softmax
merged_softmax = (softmax1 + softmax2) / 2
segmentation = merged_softmax.argmax(axis=0).astype(np.uint8)
# print(segmentation.shape)
# print(gt1.shape)

in1 = nib.load("/scratch/tang.zitian/nnUNet_predictions/sep_T1T2_predictions/BraTS2021_00016T1.nii.gz").get_fdata().astype(np.uint8)
print(in1.shape)
print(softmax1.shape)







# save
# img = nib.Nifti1Image(segmentation, np.eye(4))
# nib.save(img, out_fname)
# img2 = nib.Nifti1Image(gt1, np.eye(4))
# nib.save(img2, out_gt)