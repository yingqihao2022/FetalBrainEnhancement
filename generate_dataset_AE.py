import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nibabel as nb
import copy
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    tmp = np.nonzero(mask);
    xind = tmp[0]
    yind = tmp[1]
    zind = tmp[2]
    xmin = np.min(xind); xmax = np.max(xind)
    ymin = np.min(yind); ymax = np.max(yind)
    zmin = np.min(zind); zmax = np.max(zind)
    ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax];
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

dpRoot='/your/path'

df_train = pd.DataFrame({'index': [f'{i+1}' for i in range(8)]})

df_train.insert(loc=1, column='high_path', value=0)
df_train.insert(loc=2, column='mean_high', value=0)
df_train.insert(loc=3, column='std_high', value=0)
df_train.insert(loc=4, column='shape0', value=0)
df_train.insert(loc=5, column='shape1', value=0)
df_train.insert(loc=6, column='shape2', value=0)
df_train.insert(loc=7, column='shape3', value=0)
df_train.insert(loc=8, column='shape4', value=0)
df_train.insert(loc=9, column='shape5', value=0)
df_train.insert(loc=10, column='seg_path', value=0)

df_val = pd.DataFrame({'index': [f'{i+1}' for i in range(8)]})
df_val.insert(loc=1,column='high_path', value=0)
df_val.insert(loc=2,column='mean_high', value=0)
df_val.insert(loc=3,column='std_high', value=0)
df_val.insert(loc=4,column='shape0', value=0)
df_val.insert(loc=5,column='shape1', value=0)
df_val.insert(loc=6,column='shape2', value=0)
df_val.insert(loc=7, column='shape3', value=0)
df_val.insert(loc=8, column='shape4', value=0)
df_val.insert(loc=9, column='shape5', value=0)
df_val.insert(loc=10, column='seg_path', value=0)

index_train=0
index_val=0
for new_id in tqdm(range(1, 2)):
    if(new_id <= 320):
        data_path=dpRoot+'/real_train/'+'fb'+ str(new_id).zfill(4)
    else:
        data_path=dpRoot+'/val/' + 'fb'+ str(new_id).zfill(4)
    if not os.path.exists(data_path):  
        print('path do not exit')
        continue  
    high=nb.load(data_path+'/fb'+ str(new_id).zfill(4)+'_brain.nii.gz').get_fdata()
    high_expand= np.expand_dims(high, -1)
    mask = high_expand > 0
    high_mean,high_std= normalize_image(high_expand,high_expand,mask)
    ind_block, ind_brain = block_ind(mask,64,0)

    # if(new_id > 320):
    for crop_number in range(0,ind_block.shape[0]):
        print("val",index_val)
        df_val.loc[index_val,'high_path']=data_path+'/fb'+ str(new_id).zfill(4)+'_brain.nii.gz'
        df_val.loc[index_val,'seg_path'] =data_path+'/fb'+ str(new_id).zfill(4)+'_mask.nii.gz'
        df_val.loc[index_val,'mean_high']=high_mean
        df_val.loc[index_val,'std_high']=high_std
        df_val.loc[index_val,'shape0']=ind_block[crop_number,0]
        df_val.loc[index_val,'shape1']=ind_block[crop_number,1]
        df_val.loc[index_val,'shape2']=ind_block[crop_number,2]
        df_val.loc[index_val,'shape3']=ind_block[crop_number,3]
        df_val.loc[index_val,'shape4']=ind_block[crop_number,4]
        df_val.loc[index_val,'shape5']=ind_block[crop_number,5]
        index_val=index_val+1
# else: 
    for crop_number in range(0,ind_block.shape[0]):
        print("train",index_train)
        df_train.loc[index_train,'high_path']=data_path+'/fb'+ str(new_id).zfill(4)+'_brain.nii.gz'
        df_train.loc[index_train,'seg_path'] =data_path+'/fb'+ str(new_id).zfill(4)+'_mask.nii.gz'
        df_train.loc[index_train,'mean_high']=high_mean
        df_train.loc[index_train,'std_high']=high_std
        df_train.loc[index_train,'shape0']=ind_block[crop_number,0]
        df_train.loc[index_train,'shape1']=ind_block[crop_number,1]
        df_train.loc[index_train,'shape2']=ind_block[crop_number,2]
        df_train.loc[index_train,'shape3']=ind_block[crop_number,3]
        df_train.loc[index_train,'shape4']=ind_block[crop_number,4]
        df_train.loc[index_train,'shape5']=ind_block[crop_number,5]
        index_train=index_train+1

print(index_train,index_val)

df_val.to_json('new_single_val_AE.json')
print(df_val)
print(df_train)
df_train.to_json('new_single_train_AE.json')
