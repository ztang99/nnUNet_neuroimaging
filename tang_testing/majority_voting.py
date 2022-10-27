
import numpy as np
import os
import nibabel as nib


def bincount2D_vectorized(a):    
    N = a.max()+1
    a_offs = a + np.arange(a.shape[0])[:,None]*N
    return np.bincount(a_offs.ravel(), minlength=a.shape[0]*N).reshape(-1,N)

def majorty_voting(images):
    ## The input images must have 4 dimensions:
    ## HxWxDxnumer of channels
    assert len(images.shape) == 4
    H, W, D, F = images.shape
    images_unfolded = images.reshape(-1, F)
    counts = bincount2D_vectorized(images_unfolded)
    voted = counts.argmax(axis=1)
    voted = voted.reshape(H, W, D)

    return voted


modals = "NT1c"
path = "/scratch/tang.zitian/nnUNet_predictions/nnunet_sep400_predictions"
# path = f"/scratch/tang.zitian/nnUNet_predictions/sep_{modals}_predictions"
gt_path = "/scratch/tang.zitian/nnUNet_raw_data_base/nnUNet_raw_data/Task400_BraTS2021/labelsTs"
# get list of subject ids
all_ids = []
for f in os.listdir(f"/scratch/tang.zitian/nnUNet_predictions/sep_{modals}_predictions"):
    if ("nii.gz" in f) & (f[:15] not in all_ids):
        all_ids.append(f[:15])

# subject_id = "BraTS2021_00016"
# path = f"/scratch/tang.zitian/nnUNet_predictions/sep_{modals}_predictions"
for subject_id in all_ids:

    file1 = nib.load(os.path.join(path, subject_id + "T1.nii.gz")).get_fdata().astype(np.uint8)
    file2 = nib.load(os.path.join(path, subject_id + "T1c.nii.gz")).get_fdata().astype(np.uint8)
    file3 = nib.load(os.path.join(path, subject_id + "T2.nii.gz")).get_fdata().astype(np.uint8)
    file4 = nib.load(os.path.join(path, subject_id + "Flair.nii.gz")).get_fdata().astype(np.uint8)

    # out_path = "/scratch/tang.zitian/nnUNet_predictions/testing"

    stacked = np.stack((file1, file2, file3, file4), axis = -1)
    # print(stacked.shape)
    out = majorty_voting(stacked).astype(np.uint8)

    img = nib.Nifti1Image(out, np.eye(4))
    # print(img.shape)
    nib.save(img, os.path.join(path, f"{subject_id}combined.nii.gz"))

    # read in and save the seg mask the same as combined masks (only need to run once)
    # gt = nib.load(os.path.join(gt_path, f"{subject_id}T1.nii.gz")).get_fdata().astype(np.uint8)
    # gt_out = nib.Nifti1Image(gt, np.eye(4))
    # nib.save(gt_out, os.path.join(gt_path, f"{subject_id}gt.nii.gz"))

# try compute dice for this one
# from nnunet.inference.dice_loss_new import dice
# image_pred = nib.load(os.path.join(out_path, "test_mv_2.nii.gz")).get_fdata().astype(np.uint8)
# image_gt = nib.load(os.path.join(out_path, "test_gt.nii.gz")).get_fdata().astype(np.uint8)
# # print(image_pred.shape)
# dice = dice(image_pred, image_gt)
# print(dice)