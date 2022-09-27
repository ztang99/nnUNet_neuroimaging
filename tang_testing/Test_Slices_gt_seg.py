## This file is to visualize gt segmented images of the BraTS2021 dataset
## Zitian Tang
## 09/09/2022

# import
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib

# global variables
data_dir = "/scratch/peijie.qiu/dataset/nnUNet_raw_data_base/\
nnUNet_cropped_data/Task101_BraTS2021/gt_segmentations"

# visualize first image set in seg data
# filenames = []
# for i in range(0,5):
#     image_path = os.path.join(data_dir, "BraTS2021_0010"+str(i)+".nii.gz")
#     filenames.append(image_path)
image_path = os.path.join(data_dir, "BraTS2021_00101.nii.gz")

#===================================
# load image (3D) [X,Y,Z_slice]
img = nib.load(image_path).get_fdata()
# print(img.shape) #(240,240,155)
slices = img[:,:,::15]
# print(slices.shape) #(240,240,8)


# create image grid

# Quilt rows and columns
nRows = 1
nCols = slices.shape[-1]
# Spacers
wspace = 0.10
hspace = 0.10
# Quilt width and height
figsize_x = 15+(wspace*nCols)
figsize_y = 15*(nRows/nCols)+(hspace*nRows)

# Plot Tiles
fig, ax = plt.subplots(nrows=nRows, ncols=nCols, 
                       figsize=(figsize_x, figsize_y),
                       gridspec_kw={'wspace': wspace, 'hspace': hspace})

for r in range(nCols):
    ax[r].axis("off")
    curr_slice = slices[:,:,r]
    ax[r].imshow(np.moveaxis(curr_slice,0,-1))

plt.savefig("/home/tang.zitian/nnUNet/tang_testing/testvis_1.png")
