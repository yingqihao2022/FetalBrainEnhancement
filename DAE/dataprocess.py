from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
from block import block_ind
import cv2

def normalise_percentile(volume):  

    v = volume.reshape(-1)  
    v_nonzero = v[v > 0]  # Use only the brain foreground to calculate the quantile  

    p_99 = np.percentile(v_nonzero, 99)

    volume /= p_99

    return volume  

def process_patient(path, target_path):

    # files = sorted(os.listdir(path), key=str.lower)
    # brain_path = os.path.join(path, files[0])  
    # mask_path =  os.path.join(path, os.listdir(path)[0])
    brain = nib.load(path).get_fdata().astype(np.float32)
    # mask = nib.load(mask_path).get_fdata()

    ind_block, ind_brain = block_ind(brain)

    brain = normalise_percentile(brain)

    # brain = apply_bilateral_filter_3d(brain)

    num_block = ind_block.shape[0]

    patient_dir = Path(target_path)
    patient_dir.mkdir(parents=True, exist_ok=True)
    last_name = patient_dir.parts[-1]
    # print(last_level_name[2:])

    brain_block = [None] * num_block
    # (target_path / split).mkdir(parents=True, exist_ok=True)
    for i in range(num_block):

        depth_start = ind_block[i][0]
        depth_end = ind_block[i][1]

        height_start = ind_block[i][2]
        height_end = ind_block[i][3]

        width_start = ind_block[i][4]
        width_end = ind_block[i][5]

        brain_block[i] = brain[depth_start:depth_end + 1,
                         height_start:height_end + 1, 
                         width_start:width_end + 1]

        brain_block[i] = torch.from_numpy(brain_block[i]).float().unsqueeze(dim=0).unsqueeze(dim=0)

        if i == num_block - 1:
            np.savez_compressed(patient_dir / f"{last_name}_part{i}_{depth_start}_{depth_end}_{height_start}_{height_end}_{width_start}_{width_end}_{ind_brain[0]}_{ind_brain[1]}_{ind_brain[2]}_{ind_brain[3]}_{ind_brain[4]}_{ind_brain[5]}",
                                x = brain_block[i])
        else:
            np.savez_compressed(patient_dir / f"{last_name}_part{i}_{depth_start}_{depth_end}_{height_start}_{height_end}_{width_start}_{width_end}",
                                x = brain_block[i])


def preprocess(datapath: Path):


    targetpath = Path("/Data/test")
    targetpath.mkdir(parents=True, exist_ok=True)
    paths = sorted([f for f in os.listdir(datapath)])

    for source_path in tqdm(paths):
        full_source_path = datapath / source_path  
        target_path = os.path.join(targetpath, f"{source_path[:6]}")
        process_patient(full_source_path, target_path)
    
if __name__ == "__main__":
    
    datapath = Path("/path/to/your/data")
    preprocess(datapath)
