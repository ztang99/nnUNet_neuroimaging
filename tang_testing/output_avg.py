"""
this file contains methods/trying to have a method to 
"""

#import libraries
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt

#tryout block
input_path = "/scratch/tang.zitian/nnUNet_predictions/sep_T1T1c_predictions"
patient_name = "BraTS2021_00016"

#readin nii.gz file
modals = ["T1","T1c"]

# data = []
# for m in modals:
#     data.append(nib.load(os.path.join(input_path, patient_name+m+".nii.gz")).get_fdata().astype(np.uint8))

# output = np.zeros((data[0].shape))
# for p in range(output.shape[0]):
#     for q in range(output.shape[1]):
#         for r in range(output.shape[2]):
#             output[p,q,r] = np.sum(data[i][p,q,r] for i in range(len(modals)))/len(modals)

def output_avg(input_path, patient_name, modals):
    """
    :param input_path: 
    :param patient_name: in format BraTS2021_0xxxx
    :param modals: list of modals

    :return avg_mask: a single mask with every pixel being the average of the input list of masks
    """
    data = []
    for m in modals:
        data.append(nib.load(os.path.join(input_path, patient_name+m+".nii.gz")).get_fdata().astype(np.uint8))

    output = np.zeros((data[0].shape))
    for p in range(output.shape[0]):
        for q in range(output.shape[1]):
            for r in range(output.shape[2]):
                output[p,q,r] = np.sum([data[i][p,q,r] for i in range(len(modals))])/len(modals)
    
    return output


#plot and save to see?
# slices = output[:,:,::15]
# # print(slices.shape) #(240,240,8)


# # create image grid

# # Quilt rows and columns
# nRows = 1
# nCols = slices.shape[-1]
# # Spacers
# wspace = 0.10
# hspace = 0.10
# # Quilt width and height
# figsize_x = 15+(wspace*nCols)
# figsize_y = 15*(nRows/nCols)+(hspace*nRows)

# # Plot Tiles
# fig, ax = plt.subplots(nrows=nRows, ncols=nCols, 
#                        figsize=(figsize_x, figsize_y),
#                        gridspec_kw={'wspace': wspace, 'hspace': hspace})

# for r in range(nCols):
#     ax[r].axis("off")
#     curr_slice = slices[:,:,r]
#     ax[r].imshow(np.moveaxis(curr_slice,0,-1))

# plt.savefig("/home/tang.zitian/nnUNet/tang_testing/testcombine.png")