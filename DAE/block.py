import numpy as np
import nibabel as nib
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

# def block2brain(blocks, inds, mask):  
#     vol_brain = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2]])  
#     vol_count = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2]])  

#     for tt in np.arange(inds.shape[0]):  
#         inds_this = inds[tt, :]  
#         vol_brain[inds_this[0]:inds_this[1]+1,   
#                   inds_this[2]:inds_this[3]+1,   
#                   inds_this[4]:inds_this[5]+1] = blocks[tt]  
#         vol_count[inds_this[0]:inds_this[1]+1,   
#                   inds_this[2]:inds_this[3]+1,   
#                   inds_this[4]:inds_this[5]+1] += 1  
                  
# mask = np.zeros((200, 200, 200), dtype=bool) 

# mask[50:150, 60:140, 70:130] = 1

# nii_image = nib.load('/data/birth/lmx/work/Class_projects/course5/dataset/Fetal_Brain_dataset/GMH_IVH_data/img/pa056.nii.gz')  

# # 获取图像数据  
# image_data = nii_image.get_fdata()  

# ind_block, ind_brain = block_ind(image_data)
# print(ind_block,ind_brain)
# print(ind_block.shape)