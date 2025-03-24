import numpy as np  
import os  
import sys  
import warnings  
warnings.filterwarnings("ignore")  
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import nibabel as nib  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader  

from partialconv2d import PartialConv2d  
from model import self2self  
from tqdm import tqdm
class NiftiDataset(Dataset):  
    def __init__(self, file_path):  
        self.nii_obj = nib.load(file_path)  
        vol = self.nii_obj.get_fdata() 
        vol = np.transpose(vol, (2, 1, 0))   
        self.vol = np.expand_dims(vol, axis=1)  

    def __len__(self):  
        return self.vol.shape[0]  

    def __getitem__(self, idx):  
        slice_3d = self.vol[idx]    # (1, H, D)  
        return torch.FloatTensor(slice_3d)  
class NewNiftiDataset(Dataset):  
    def __init__(self, directory_path):  
        import os  
        import re  
        
        # Find all matching files with fb0391 and onwards  
        self.file_paths = []  
        self.file_ids = []  
        
        pattern = re.compile(r'fb(\d+)_brain\.nii\.gz')  
        
        for filename in os.listdir(directory_path):  
            match = pattern.match(filename)  
            if match:  
                file_num = int(match.group(1))  
                if file_num >= 391:  # Only include fb0391 and onwards  
                    self.file_paths.append(os.path.join(directory_path, filename))  
                    self.file_ids.append(f"fb{file_num:04d}")  
        
        # Sort files to ensure consistent ordering  
        sorted_pairs = sorted(zip(self.file_paths, self.file_ids), key=lambda x: x[1])  
        self.file_paths, self.file_ids = zip(*sorted_pairs) if sorted_pairs else ([], [])  
        
        # Get dimensions of each file  
        self.slices_per_file = []  
        for file_path in self.file_paths:  
            nii_obj = nib.load(file_path)  
            shape = nii_obj.shape  
            self.slices_per_file.append(shape[2])  # W dimension in original (D, H, W)  
        
        # Calculate cumulative slices for indexing  
        self.cumulative_slices = [0]  
        for count in self.slices_per_file:  
            self.cumulative_slices.append(self.cumulative_slices[-1] + count)  
    
    def __len__(self):  
        return self.cumulative_slices[-1] if self.cumulative_slices else 0  
    
    def __getitem__(self, idx):  
        # Find which file and which slice  
        file_idx = 0  
        while file_idx < len(self.cumulative_slices) - 1 and idx >= self.cumulative_slices[file_idx + 1]:  
            file_idx += 1  
        
        # Calculate the slice index within the file  
        slice_idx = idx - self.cumulative_slices[file_idx]  
        
        # Load the file at the given index  
        nii_obj = nib.load(self.file_paths[file_idx])  
        vol = nii_obj.get_fdata()  
        vol = np.transpose(vol, (2, 1, 0))  
        vol = np.expand_dims(vol, axis=1)  
        
        # Get the specific slice  
        slice_3d = vol[slice_idx]  # (1, H, D)  
        file_id = self.file_ids[file_idx]  
        
        return torch.FloatTensor(slice_3d), file_id  

def pad_h_and_w(tensor, desired_size=512):  
    _, _, H, W = tensor.shape   
    pad_h_total = max(desired_size - H, 0)  
    pad_w_total = max(desired_size - W, 0)  

    pad_top = pad_h_total // 2  
    pad_bottom = pad_h_total - pad_top  
    pad_left = pad_w_total // 2  
    pad_right = pad_w_total - pad_left  

    tensor_padded = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))  
    return tensor_padded, (pad_top, pad_bottom, pad_left, pad_right)  

def unpad_h_and_w(tensor, pad_info):  
    pad_top, pad_bottom, pad_left, pad_right = pad_info  
    _, _, H, W = tensor.shape  
    unpad_h = slice(pad_top, H - pad_bottom if pad_bottom > 0 else None)  
    unpad_w = slice(pad_left, W - pad_right if pad_right > 0 else None)  
    return tensor[:, :, unpad_h, unpad_w]  

if __name__ == "__main__":  
    USE_GPU = True  
    dtype = torch.float32  
    device = torch.device('cuda') if USE_GPU and torch.cuda.is_available() else torch.device('cpu')  
    print('using device:', device)  
    model = self2self(1, 0.3).to(device)  
    optimizer = optim.Adam(model.parameters(), lr=1e-4)   
    dataset_path = "/data/birth/lmx/work/Class_projects/course5/dataset/Fetal_Brain_dataset/Normal_data/img"  
    dataset = NiftiDataset(dataset_path)  
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  
    p = 0.3  
    NPred = 100  
    max_iters = 500000  
    original_nii = dataset.nii_obj  
    original_affine = original_nii.affine  
    original_header = original_nii.header  
    
    for itr in range(max_iters):  
        for batch_data in dataloader:   
            if torch.all(batch_data == 0):  
                continue  

            batch_data = batch_data.to(device)  
            model.train()  

            brain_mask = (batch_data > 0).float()  
            batch_data_padded, pad_info = pad_h_and_w(batch_data, desired_size=512)  
            p_mtx = torch.rand_like(batch_data_padded)  
            drop_mask = (p_mtx > p).float() * 0.7   
            img_input = batch_data_padded * drop_mask  
            output_padded = model(img_input, drop_mask)   
            output_unpadded = unpad_h_and_w(output_padded, pad_info)  
            drop_mask_unpad = unpad_h_and_w(drop_mask, pad_info)   
            combined_mask = (1 - drop_mask_unpad) * brain_mask  
            if torch.sum(combined_mask) < 1e-6:  
                continue  

            loss = torch.sum((output_unpadded - batch_data)**2 * combined_mask) / torch.sum(combined_mask)  

            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()    
            break  

        print(f"Iteration {itr+1}, loss = {loss.item()*100:.4f}")  
        if (itr + 1) % 1000 == 0:  
            model.eval()  
            vol_shape = dataset.vol.shape  # (W, 1, H, D)  
            sum_preds = np.zeros((vol_shape[0], vol_shape[2], vol_shape[3]), dtype=np.float32)  

            with torch.no_grad():  
                for _ in tqdm(range(NPred)):  
                    pred_idx = 0  
                    for d in range(vol_shape[0]):  
                        slice_data = dataset[d].unsqueeze(0)  # (1, 1, H, W)  
                        # 跳过全0切片，但在结果中要占位  
                        if torch.all(slice_data == 0):  
                            pred_idx += 1  
                            continue  

                        slice_data = slice_data.to(device)   
                        brain_mask_test = (slice_data > 0).float()  
                        slice_data_padded, pad_info = pad_h_and_w(slice_data, desired_size=512)  
                        p_mtx = torch.rand_like(slice_data_padded)  
                        mask_test = (p_mtx > p).float() * 0.7  
                        out_test_padded = model(slice_data_padded * mask_test, mask_test)   
                        out_test_unpad = unpad_h_and_w(out_test_padded, pad_info)  
                        out_test_unpad = out_test_unpad * brain_mask_test  
                        out_np = out_test_unpad.cpu().numpy()[0, 0]  # (H, W)  
                        sum_preds[pred_idx] += out_np  
                        pred_idx += 1  

            avg_preds = sum_preds / float(NPred)  
            norm_min, norm_max = avg_preds.min(), avg_preds.max()  
            avg_preds_3d = np.expand_dims(avg_preds, axis=1)  
            avg_preds_3d = np.squeeze(avg_preds_3d, axis=1)  # (W, H, D)  
            avg_preds_3d = np.transpose(avg_preds_3d, (2, 1, 0))  # (D, H, W)  
  
            out_nii = nib.Nifti1Image(avg_preds_3d, affine=original_affine, header=original_header)  
            nib.save(out_nii, f"self2self3d/image/3d_output_{itr+1}.nii.gz")  

            #torch.save(model.state_dict(), f"model_{itr+1}.pth")
