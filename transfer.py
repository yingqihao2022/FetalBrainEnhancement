from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F


def process(data_path, mask_path = None, use_mask = True):
    brain = nib.load(data_path).get_fdata()
    if use_mask:
        mask = nib.load(mask_path).get_fdata()
    else:
        mask = (brain != 0)

    matrix = np.full((210, 210, 210), np.min(brain))
    matrix = np.full((135, 189, 155), np.min(brain))
    brain = brain - matrix
    brain = brain * mask

    brain_padding = np.zeros((210,210,210))
    brain_padding[38:173, 10:199, 28:183] += brain
    brain_save = nib.Nifti1Image(brain_padding, affine=np.eye(4)) 

    output_dir = data_path
    output_path = output_dir / f"{data_path}" 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(brain_save, output_path)


datapath = Path("/data/birth/lmx/work/Class_projects/course5/work/vae_hj/new_output/38w")
maskpath = Path("/data/birth/lmx/work/Class_projects/course5/dataset/Fetal_Brain_dataset/GMH_IVH_data/mask") # optional 

datapaths = sorted([datapath / f for f in os.listdir(datapath)])
maskpaths = sorted([maskpath / f for f in os.listdir(maskpath)])
print(len(datapaths), len(maskpaths))
# assert len(datapaths) == len(maskpaths)
for idx in tqdm(range(len(datapaths))):
    process(datapaths[idx], maskpaths[idx], use_mask = False)
