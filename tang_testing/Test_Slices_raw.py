## This file is to visualize the raw images in the BraTS2021 dataset
## Zitian Tang
## 09/09/2022

"""
Important information regarding dimension of the MRI images:
Shape follows: (Modality, H,  W, D)
, where Modality: 
0 - T1, 1 - T1c, 2 - T2, 3 - FLAIR, 4 - segmentation mask
"""

# import
import numpy as np
import matplotlib.pyplot as plt
import os

# global variables
data_dir = "/scratch/peijie.qiu/dataset/nnUNet_raw_data_base/\
nnUNet_cropped_data/Task101_BraTS2021"

image_path = os.path.join(data_dir, "BraTS2021_00101.npz")

# visualize second image in BraTS2021 dataset (raw)
image = np.load(image_path)
# for k in image.keys():
#     print(k) #data
# print(len(image.keys()))
fdata = image["data"]
# print(fdata.shape) #(5, 137, 178, 137)

# get data for each modality
fdata_T1 = fdata[0,:,:,:]
fdata_T1c = fdata[1,:,:,:]
fdata_T2 = fdata[2,:,:,:]
fdata_FLAIR = fdata[3,:,:,:]
seg_mask = fdata[4,:,:,:]
# print(fdata_FLAIR.shape) # (137, 178, 137)

#===============================
# create image grid
slice_num = 50
titles = ["T1", "T1c", "T2", "FLAIR", "Segmentation Mask"]
# Quilt rows and columns
nRows = 1
nCols = 5 # 5 modalities
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

ax[0].imshow(np.moveaxis(fdata_T1[slice_num],0,-1)) # (137[1], 178, 137[2]) --> (178, 137[2], 137[1])
ax[1].imshow(np.moveaxis(fdata_T1c[slice_num],0,-1))
ax[2].imshow(np.moveaxis(fdata_T2[slice_num],0,-1))
ax[3].imshow(np.moveaxis(fdata_FLAIR[slice_num],0,-1))
ax[4].imshow(np.moveaxis(seg_mask[slice_num],0,-1))

for r in range(nCols):
    ax[r].axis("off")
    ax[r].set_title(titles[r])

plt.savefig("/home/tang.zitian/nnUNet/tang_testing/testmod_2.png")