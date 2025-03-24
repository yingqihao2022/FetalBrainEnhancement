from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import os  
from torch.utils.data import Dataset, Subset    
import warnings
import torch.nn as nn
from math import sqrt
from torch.nn import functional as F
import torch  
import random
from tqdm import tqdm
import re
warnings.filterwarnings("ignore")
import nibabel as nib

class FetalBrainDataset(Dataset):  
    def __init__(self, data_dir, split = "train"):  

        self.data_dir = Path(data_dir)  
        self.split = split
        self.data_files = self._collect_data_files(split)  
        self.len = len(self.data_files) 
        self.idx_map = {x: x for x in range(self.len)}
    def _collect_data_files(self, split):  

        data_files = []  

        for patient_dir in sorted(self.data_dir.iterdir()):  
            if patient_dir.is_dir():  
                if(self.split == "train"):
                    patient_files = sorted(patient_dir.glob('*part*.npz'),
                                           key=lambda x: int(x.stem.split('_part')[-1]))
                else: 
                    assert self.split == "test"

                    patient_files = sorted(patient_dir.glob('*_part*.npz'),   
                    key=lambda x: int(x.stem.split('_part')[1].split('_')[0]))   
                data_files.extend(patient_files)  

        return data_files  
    
    def __len__(self):  
        return self.len
    
    def __getitem__(self, idx):  
        idx = self.idx_map[idx]
        data = np.load(self.data_files[idx])

        if(self.split == 'train'):
            data = data[data.files[0]].squeeze(0)  
            # data = torch.from_numpy(data).float()
            return data
        elif(self.split == 'test'):
            # data = process(**data)
            # data = (data[data.files[0]].squeeze(0), data[data.files[1]].squeeze(0))
            data = data[data.files[0]].squeeze(0)
            file_name = os.path.basename(self.data_files[idx])  
            return data, file_name

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class CustomSwish(nn.Module):
    def forward(self, input_tensor):
        return Swish.apply(input_tensor)
    
def get_groups(channels: int) -> int:
    """
    :param channels:
    :return: return a suitable parameter for number of groups in GroupNormalisation'.
    """
    divisors = []
    for i in range(1, int(sqrt(channels)) + 1):
        if channels % i == 0:
            divisors.append(i)
            other = channels // i
            if i != other:
                divisors.append(other)
    return sorted(divisors)[len(divisors) // 2]

class UNet3D(nn.Module):
    def __init__(
            self,
            in_channels=1,
            n_classes=1,   
            depth=4,
            wf=6,
            padding=True,
            ):
        super(UNet3D, self).__init__()
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels

        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock3D(prev_channels, 2 ** (wf + i), padding)  # new convblock is waiting to be redifined.
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock3D(prev_channels, 2 ** (wf + i), padding) # new upconvblock is waiting to be redifined.
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv3d(prev_channels, n_classes, kernel_size = 1)  # No change here
 
    def forward_down(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            if i != len(self.down_path) - 1:
                x = F.avg_pool3d(x, 2)    # No change here

        return x, blocks

    def forward_up_without_last(self, x, blocks):
        for i, up in enumerate(self.up_path):
            skip = blocks[-i - 2]
            x = up(x, skip)

        return x

    def forward_without_last(self, x):
        x, blocks = self.forward_down(x)
        x = self.forward_up_without_last(x, blocks)
        return x

    def forward(self, x):
        x = self.get_features(x)
        return self.last(x)

    def get_features(self, x):
        return self.forward_without_last(x)

    # From here needs to be changed (Two new classes)
    
class UNetConvBlock3D(nn.Module):
    def __init__(self, in_size, out_size, padding, kernel_size=3):
        super(UNetConvBlock3D, self).__init__()
        block = []
        if padding:
            block.append(nn.ReflectionPad3d(1))
        block.append(nn.Conv3d(in_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())
        block.append(nn.GroupNorm(get_groups(out_size), out_size))

        if padding:
            block.append(nn.ReflectionPad3d(1))
        block.append(nn.Conv3d(out_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())
        block.append(nn.GroupNorm(get_groups(out_size), out_size))  
        self.block = nn.Sequential(*block)
    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock3D(nn.Module):
    def __init__(self, in_size, out_size, padding):
        super(UNetUpBlock3D, self).__init__()

        self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2)
        self.conv_block = UNetConvBlock3D(in_size, out_size, padding)
        
    def forward(self, x, bridge):   # confirmation needed
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D().to(device)
    checkpoint_path = '/path/to/your/ckpt/path'  
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'Checkpoint loaded: {checkpoint_path}, epoch: {epoch}, loss: {loss}')
    else:
        print(f'Checkpoint not found: {checkpoint_path}')
    test_datapath = 'path/to/your/data'
    test_dataset = FetalBrainDataset(test_datapath, split = "test")
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,drop_last=True,num_workers=6)
    output_dir = '/path/to/your/output_dir'
    for batch, name in tqdm(test_dataloader):
        
        x = batch.to(device)
        numbers = re.findall(r'\d+', name[0])
        output = model(x)  
        # '0002_part0_55_118_50_113_58_121.npz', this is a normal input
        if(len(numbers) == 8):
            index = (numbers[2], numbers[3], numbers[4], numbers[5], numbers[6], numbers[7])
            index = np.array(list(int(num) for num in index))  
            index_list.append(index)
            output_list.append(output.cpu())
            break

        else:
            date_part = name[0].split('_')[0] 
            new_name = f"pa{date_part}"  
            index = (numbers[2], numbers[3], numbers[4], numbers[5], numbers[6], numbers[7])
            index = np.array(list(int(num) for num in index))  
            index_list.append(index)
            output_list.append(output.cpu())
            index_all = (int(numbers[8]), int(numbers[10]), int(numbers[12]))
            index_all_max = (int(numbers[9]) - int(numbers[8]) + 1, int(numbers[11]) - int(numbers[10]) + 1, int(numbers[13]) - int(numbers[12]) + 1)
            output_res = np.zeros(shape = index_all_max)
            record_res = np.zeros(shape = index_all_max)

            for i in range(len(output_list)):
                x_start = index_list[i][0] - index_all[0]  
                x_end = index_list[i][1] + 1 - index_all[0]  
                y_start = index_list[i][2] - index_all[1]  
                y_end = index_list[i][3] + 1 - index_all[1]  
                z_start = index_list[i][4] - index_all[2]  
                z_end = index_list[i][5] + 1 - index_all[2]  
                output_res[index_list[i][0] - index_all[0] : index_list[i][1] + 1 - index_all[0],
                            index_list[i][2] - index_all[1] : index_list[i][3] + 1 - index_all[1],
                            index_list[i][4] - index_all[2] : index_list[i][5] + 1 - index_all[2],
                            ] += output_list[i].numpy().squeeze(0).squeeze(0)
                record_res[index_list[i][0] - index_all[0] : index_list[i][1] + 1 - index_all[0],
                            index_list[i][2] - index_all[1] : index_list[i][3] + 1 - index_all[1],
                            index_list[i][4] - index_all[2] : index_list[i][5] + 1 - index_all[2],
                            ] += 1
            index_list = []
            output_list = []   
            output_res /= record_res
            padded_output = np.zeros((210, 210, 210))  
            padded_output[int(numbers[8]):int(numbers[9])+1,   
                        int(numbers[10]):int(numbers[11])+1,   
                        int(numbers[12]):int(numbers[13])+1] = output_res  
            nifti_img_output = nib.Nifti1Image(padded_output, affine=np.eye(4)) 
            output_path = output_dir / f"{new_name}.daedenoised.nii.gz" 
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            nib.save(nifti_img_output, output_path)
