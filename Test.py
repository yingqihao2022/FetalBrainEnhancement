import numpy as np
import torch
from vaedataset import *
import nibabel as nb
import pandas as pd
import copy
from tqdm import tqdm
#model
import sys
sys.path.append('/data/birth/lmx/work/Class_projects/course5/dataset/Fetal_Brain_dataset/VAE_denoise')
from autoencoderkl import AutoencoderKL
import os


torch.cuda.set_device(5) 

def normalize_image(imgall, imgresall, mask, norm_ch='all'):
    imgall_norm = copy.deepcopy(imgall)
    imgresall_norm = copy.deepcopy(imgresall)
    if norm_ch == 'all':
        norm_ch = np.arange(imgall.shape[-1])
    for jj in norm_ch:
        img = imgall[:, :, :, jj : jj + 1]
        imgres = imgresall[:, :, :, jj : jj + 1]
        img_mean = np.mean(img[mask > 0.5])
        img_std = np.std(img[mask > 0.5])
        #img_norm = (img - img_mean) / img_std * mask
        #imgres_norm = (imgres - img_mean) / img_std * mask
        #imgall_norm[:, :, :, jj : jj + 1] = img_norm
        #imgresall_norm[:, :, :, jj : jj + 1] = imgres_norm
    #return imgall_norm, imgresall_norm
    return img_mean,img_std
def block_ind(mask, sz_block=64, sz_pad=0):
    # find indices of smallest block that covers whole brain
    tmp = np.nonzero(mask)
    xind = tmp[0]
    yind = tmp[1]
    zind = tmp[2]
    xmin = np.min(xind); xmax = np.max(xind)
    ymin = np.min(yind); ymax = np.max(yind)
    zmin = np.min(zind); zmax = np.max(zind)
    ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax]
    # calculate number of blocks along each dimension
    xlen = xmax - xmin + 1
    ylen = ymax - ymin + 1
    zlen = zmax - zmin + 1
    nx = int(np.ceil(xlen / sz_block)) + sz_pad
    ny = int(np.ceil(ylen / sz_block)) + sz_pad
    nz = int(np.ceil(zlen / sz_block)) + sz_pad
    # determine starting and ending indices of each block
    xstart = xmin
    ystart = ymin
    zstart = zmin
    xend = xmax - sz_block + 1
    yend = ymax - sz_block + 1
    zend = zmax - sz_block + 1
    xind_block = np.round(np.linspace(xstart, xend, nx))
    yind_block = np.round(np.linspace(ystart, yend, ny))
    zind_block = np.round(np.linspace(zstart, zend, nz))
    ind_block = np.zeros([xind_block.shape[0]*yind_block.shape[0]*zind_block.shape[0], 6])
    count = 0
    for ii in np.arange(0, xind_block.shape[0]):
        for jj in np.arange(0, yind_block.shape[0]):
            for kk in np.arange(0, zind_block.shape[0]):
                ind_block[count, :] = np.array([xind_block[ii], xind_block[ii]+sz_block-1, yind_block[jj], yind_block[jj]+sz_block-1, zind_block[kk], zind_block[kk]+sz_block-1])
                count = count + 1
    ind_block = ind_block.astype(int)
    return ind_block, ind_brain
def block2brain(blocks, inds, mask):
    vol_brain = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2]])
    vol_count = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2]])
    
    for tt in np.arange(inds.shape[0]):
        inds_this = inds[tt, :]
        
        vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1] = \
                vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1] + blocks[tt, :, :, :]
        
        vol_count[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1] = \
                vol_count[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1] + 1.
    
    vol_count[vol_count < 0.5] = 1.
    vol_brain = vol_brain / vol_count 
    
    vol_brain = vol_brain * mask
    vol_count = vol_count * mask
    
    return vol_brain, vol_count 


dpRoot='/data/birth/lmx/data/datasets/huaxi_alats/37w/'
# Function to list all file names in a given directory

def list_files_in_directory(directory_path):

    file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    return file_names
new_id_list=list_files_in_directory(dpRoot)


# def list_files_in_directory(directory_path, start_id, end_id):   
#     file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]  
#     filtered_files = []  
#     for file_name in file_names:  
#         if file_name.startswith("pa") and file_name.endswith(".nii.gz"):  
#             try:  
#                 file_id = int(file_name[2:5])  
#                 if start_id <= file_id <= end_id:  
#                     filtered_files.append(file_name)  
#             except ValueError:  
#                 continue  
    
#     return filtered_files  
# start_id = 55
# end_id = 58
# new_id_list = list_files_in_directory(dpRoot, start_id, end_id)  
# print("Number of files:", len(new_id_list))  


for new_id in tqdm(new_id_list):
    data_path=dpRoot+new_id
    high=nb.load(data_path)
    img_affine=high.affine
    high=high.get_fdata()
    high_copy=high.copy()
    high_expand= np.expand_dims(high, -1)
    mask = high_expand > 0
    
    high_mean,high_std= normalize_image(high_expand,high_expand,mask)
    
    ind_block, ind_brain = block_ind(mask,64,0)

    high[high>0]= (high[high>0]- high_mean) / high_std

    # age=(age-140)/160
    # age = np.full((1, 1, 64, 64, 64), age)
    # age = torch.tensor(age, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net= AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(64, 128, 192),
    latent_channels=3,
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True),
    ).to(device)
    net.load_state_dict(torch.load("/data/birth/lmx/work/Class_projects/course5/work/vae_hj/ckpt_390_norm/autoencoder_minloss.pth")['net'])
    net.eval()

    blocks=np.zeros([ind_block.shape[0],64,64,64])
    for ii in range(ind_block.shape[0]):
        high_block=high[ind_block[ii,0]:ind_block[ii,1]+1,ind_block[ii,2]:ind_block[ii,3]+1,ind_block[ii,4]:ind_block[ii,5]+1]
        high_copy_block=high_copy[ind_block[ii,0]:ind_block[ii,1]+1,ind_block[ii,2]:ind_block[ii,3]+1,ind_block[ii,4]:ind_block[ii,5]+1]
        high_block = high_block.reshape((1,1,)+high_block.shape)
        high_block = torch.tensor(high_block, dtype=torch.float32)
        with torch.no_grad():
           output,a,b = net(high_block.to(device))
        
        # if ii==1:
        #     eps = torch.randn_like(b)
        #     z_vae = a + eps * b
        #     z_vae=z_vae.to('cpu').numpy().reshape(4,16,16,16)
        #     nb.Nifti1Image(z_vae[0,:,:,:],img_affine).to_filename('/data/birth/yhj/fetal_SR/LDM_0727/autoencoder_patch/latent0.nii.gz')
        #     nb.Nifti1Image(z_vae[1,:,:,:],img_affine).to_filename('/data/birth/yhj/fetal_SR/LDM_0727/autoencoder_patch/latent1.nii.gz')
        #     nb.Nifti1Image(z_vae[2,:,:,:],img_affine).to_filename('/data/birth/yhj/fetal_SR/LDM_0727/autoencoder_patch/latent2.nii.gz')
        #     nb.Nifti1Image(z_vae[3,:,:,:],img_affine).to_filename('/data/birth/yhj/fetal_SR/LDM_0727/autoencoder_patch/latent3.nii.gz')

        output=output.to('cpu').numpy().reshape(64,64,64)
        #output[high_copy_block>0]=output[high_copy_block>0]*high_std+high_mean
        blocks[ii,:]=output
    vol_brain, vol_count = block2brain(blocks,ind_block,high_copy>0)
    
    nb.Nifti1Image(vol_brain,img_affine).to_filename('/data/birth/lmx/work/Class_projects/course5/work/vae_hj/new_output/37w/'+new_id)