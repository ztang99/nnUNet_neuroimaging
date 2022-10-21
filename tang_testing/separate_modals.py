# """
# This file separates modalities from nnunet preprocessed data and save them as .npy files.
# Also checks and copies over the .pkl files and modify them.
# """

import numpy as np
import os
from collections import OrderedDict

from batchgenerators.utilities.file_and_folder_operations import *

# input_path = "/scratch/tang.zitian/nnUNet_preprocessed/Task200_BraTS2021/nnUNetData_plans_v2.1_verybig_stage0"
input_path = "/scratch/tang.zitian/nnUNet_preprocessed/Task200_BraTS2021"
output_path = "/scratch/tang.zitian/nnUNet_preprocessed/"

# ## For all .npy file in input_path, read them in, save the corresponding layers to output_path
# if len(os.listdir(output_path)) == 0:
#     for i in os.listdir(input_path):
#         if ".npy" in i: # total 938 .npy files
#             data = np.load(os.path.join(input_path, i))
#             t1 = data[(1,-1),...]
#             t1c = data[(2,-1),...]
#             t2 = data[(3,-1),...]
#             fl = data[(4,-1),...]
#             # save them to output_path
#             np.save(os.path.join(output_path, i[:-4]+"_0000.npy"), t1)
#             np.save(os.path.join(output_path, i[:-4]+"_0001.npy"), t1c)
#             np.save(os.path.join(output_path, i[:-4]+"_0002.npy"), t2)
#             np.save(os.path.join(output_path, i[:-4]+"_0003.npy"), fl)
#     print("done.")

#     # Check the shape of the data
#     p = os.listdir(output_path)[0]
#     print(np.load(os.path.join(output_path,p)).shape)

# def mult_by_4(list):
#     new_list = []
#     for i in list:
#         for j in range(0,4):
#             new_list.append(i)
#     return new_list
    

# # Modify and save the .pkl files
# """
# keys for dataset_properties.pkl: 
# > [changed] all_sizes: len=938 
# > [changed] all_spacings: len=938
# all_classes: [1,2,3]
# modalities: {0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'FLAIR'} 
# intensityproperties: None
# > [changed] size_reductions: len=938
# """

# dp = load_pickle(os.path.join(input_path, "dataset_properties.pkl"))
# dp_data = dp["all_sizes"]
# new_dp = mult_by_4(dp_data)
# dp["all_sizes"] = new_dp
# sp_data = dp["all_spacings"]
# new_sp = mult_by_4(sp_data)
# dp["all_spacings"] = new_sp
# size_data = dp["size_reductions"]
# new_size = mult_by_4(size_data)
# dp["size_reductions"] = new_size

# write_pickle(dp, os.path.join(output_path, "dataset_properties.pkl"))


# """
# keys for nnUNetPlansv2.1_verybig_plans_3D.pkl:
# num_stages: 1
# num_modalities: 4
# modalities: {0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'FLAIR'}
# normalization_schemes: OrderedDict([(0, 'nonCT'), (1, 'nonCT'), (2, 'nonCT'), (3, 'nonCT')])
# > [changed] dataset_properties: len=6 (just what's inside dataset_properties.pkl)
# > list_of_npz_files: len=938 (we don't have npz files now)
# > [changed] original_spacings: len=938, same as all_spacings in dataset_properties.pkl
# > [changed]original_sizes: len=938, same as all_sizes in dataset_properties.pkl
# > [changed] preprocessed_data_folder: /scratch/tang.zitian/nnUNet_preprocessed/Task200_BraTS2021
# num_classes: 3
# all_classes: [1,2,3], same as all_classes in dataset_properties.pkl
# base_num_features: 32
# use_mask_for_norm: OrderedDict([(0, True), (1, True), (2, True), (3, True)])
# keep_only_largest_region: None
# min_region_size_per_class: None
# min_size_per_class: None
# transpose_forward: [0,1,2]
# transpose_backward: [0,1,2]
# data_identifier: nnUNetData_plans_v2.1_verybig
# plans_per_stage: 
# {0: {'batch_size': 3, 'num_pool_per_axis': [5, 5, 5], 
# 'patch_size': array([160, 192, 160]), 'median_patient_size_in_voxels': array([140, 171, 137]), 
# 'current_spacing': array([1., 1., 1.]), 'original_spacing': array([1., 1., 1.]), 'do_dummy_2D_data_aug': False, 
# 'pool_op_kernel_sizes': [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]], 
# 'conv_kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]}}
# preprocessor_name: GenericPreprocessor
# conv_per_stage: 2
# """

# p3d = load_pickle(os.path.join(input_path, "nnUNetPlansv2.1_verybig_plans_3D.pkl"))
# dp_data = p3d["original_sizes"]
# new_dp = mult_by_4(dp_data)
# p3d["original_sizes"] = new_dp
# sp_data = p3d["original_spacings"]
# new_sp = mult_by_4(sp_data)
# p3d["original_spacings"] = new_sp
# dp = load_pickle(os.path.join(output_path, "dataset_properties.pkl"))
# p3d["dataset_properties"] = dp
# p3d["preprocessed_data_folder"] = "/scratch/tang.zitian/nnUNet_preprocessed/Task300_BraTS2021_sepmodals"
# write_pickle(p3d, os.path.join(output_path, "nnUNetPlansv2.1_verybig_plans_3D.pkl"))


# """
# keys for dataset.json:
# description: nothing
# labels: {'0': 'background', '1': 'edema', '2': 'non-enhancing', '3': 'enhancing'}
# licence: see BraTS2020 license
# modality: {'0': 'T1', '1': 'T1ce', '2': 'T2', '3': 'FLAIR'}
# name: BraTS2020
# numTest: 0
# > [changed] numTraining: 938
# reference: see BraTS2020
# release: 0.0
# tensorImageSize: 4D
# test: []
# training: []
# """

# ds = load_json(os.path.join(input_path, "dataset.json"))
# ds["numTraining"] = 3752
# save_json(ds, os.path.join(output_path, "dataset.json"))


# """
# keys for splits_final.pkl:
# len=5, in each split[i] file, there are two keys: "train" and "val"
# "train" sets len: 750,750,750,751,751
# "val" sets len: 188,188,188,187,187
# """
def mod_split(list):
    new_list = []
    for i in list:
        new_list.append(i+"T1")
        new_list.append(i+"T1c")
        new_list.append(i+"T2")
        new_list.append(i+"Flair")
        # print(i)
    return new_list

split = load_pickle(os.path.join(input_path, "splits_final.pkl"))
split[0]["train"] = mod_split(split[0]["train"])
split[1]["train"] = mod_split(split[1]["train"])
split[2]["train"] = mod_split(split[2]["train"])
split[3]["train"] = mod_split(split[3]["train"])
split[4]["train"] = mod_split(split[4]["train"])
split[0]["val"] = mod_split(split[0]["val"])
split[1]["val"] = mod_split(split[1]["val"])
split[2]["val"] = mod_split(split[2]["val"])
split[3]["val"] = mod_split(split[3]["val"])
split[4]["val"] = mod_split(split[4]["val"])
print(split[0]["train"])
# print(split[0]["val"])


write_pickle(split, os.path.join(output_path, "splits_final.pkl"))
