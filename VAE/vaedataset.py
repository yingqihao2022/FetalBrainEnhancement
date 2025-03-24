import numpy as np
import torch
import nibabel as nb
import pandas as pd
import random

def random_flip_3d(image):
    if random.random() > 0.5:
        axis = random.choice([0, 1])
        image = np.flip(image, axis).copy()
    return image

class UKBDataset(torch.utils.data.Dataset):
  def __init__(self, setname):
    self.setname=setname
    if setname=='train':
        t1_data=pd.read_json("/train_AE.json", dtype={'high_path': str, 'seg_path': str})
    elif setname=='val':
        t1_data=pd.read_json("/val_AE.json", dtype={'high_path': str, 'seg_path': str})
    self.t1_data = t1_data

  def __len__(self):
    return len(self.t1_data)
  
  def __getitem__(self, index):
    x1=self.t1_data.iat[index,4]
    x2=self.t1_data.iat[index,5]+1
    y1=self.t1_data.iat[index,6]
    y2=self.t1_data.iat[index,7]+1
    z1=self.t1_data.iat[index,8]
    z2=self.t1_data.iat[index,9]+1
    
    T2high=nb.load((self.t1_data.iat[index,1])).get_fdata()
    T2high[T2high>0]= (T2high[T2high>0]- self.t1_data.iat[index,2]) / self.t1_data.iat[index,3]
    shift_num_1=random.choice([-3,-2,-1,0,1,2,3])
    shift_num_2=random.choice([-3,-2,-1,0,1,2,3])
    shift_num_3=random.choice([-3,-2,-1,0,1,2,3])
    T2high_block=T2high[x1+shift_num_1:x2+shift_num_1,y1+shift_num_2:y2+shift_num_2,z1+shift_num_3:z2+shift_num_3]
    T2high_block=random_flip_3d(T2high_block)
    T2high_block = T2high_block.reshape((1,)+T2high_block.shape)
    T2high_block = torch.tensor(T2high_block, dtype=torch.float32)

    data_pair = {'high': T2high_block}
    return data_pair
