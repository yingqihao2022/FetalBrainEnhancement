import numpy as np  
import os  
import sys  
import warnings  
warnings.filterwarnings("ignore")  
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import nibabel as nib  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader  
from partialconv2d import PartialConv2d  
from model import self2self  
from tqdm import tqdm
import os  
import re 

class NiftiDataset(Dataset):  
    def __init__(self, file_path):  
        # 读取原始nii，并保留头信息  
        self.nii_obj = nib.load(file_path)  
        vol = self.nii_obj.get_fdata()  # 原始形状 (D, H, W)  
        # 转置为 (W, H, D)  
        vol = np.transpose(vol, (2, 1, 0))  
        
        # 存为 (W, 1, H, D)  
        self.vol = np.expand_dims(vol, axis=1)  

    def __len__(self):  
        return self.vol.shape[0]  

    def __getitem__(self, idx):  
        slice_3d = self.vol[idx]    # (1, H, D)  
        return torch.FloatTensor(slice_3d)  

def pad_h_and_w(tensor, desired_size=512):  
    _, _, H, W = tensor.shape  

    # 计算需要填充的量  
    pad_h_total = max(desired_size - H, 0)  
    pad_w_total = max(desired_size - W, 0)  

    pad_top = pad_h_total // 2  
    pad_bottom = pad_h_total - pad_top  
    pad_left = pad_w_total // 2  
    pad_right = pad_w_total - pad_left  

    # F.pad 的参数顺序是 (left, right, top, bottom)  
    tensor_padded = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))  
    return tensor_padded, (pad_top, pad_bottom, pad_left, pad_right)  

def unpad_h_and_w(tensor, pad_info):  
    pad_top, pad_bottom, pad_left, pad_right = pad_info  
    _, _, H, W = tensor.shape  

    # 先去掉上下的pad  
    # 在索引时, 维度顺序 (B, C, H, W), 因此 H 维索引用 [:, :, top : H - bottom]  
    unpad_h = slice(pad_top, H - pad_bottom if pad_bottom > 0 else None)  
    unpad_w = slice(pad_left, W - pad_right if pad_right > 0 else None)  
    return tensor[:, :, unpad_h, unpad_w]  

def train_self2self_for_file(file_path, file_id, max_iters=500000):  
    print(f"\n=== 开始训练文件: {file_id} ===")  
    
    USE_GPU = True  
    dtype = torch.float32  
    device = torch.device('cuda') if USE_GPU and torch.cuda.is_available() else torch.device('cpu')  
    print('using device:', device)  

    # 为每个文件创建一个新模型  
    model = self2self(1, 0.3).to(device)  
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  

    # 为单个文件创建数据集  
    dataset = NiftiDataset(file_path)  
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  

    # 超参数  
    p = 0.3  
    NPred = 100  
    
    # 从dataset中获取头文件和affine以在保存时保持一致  
    original_nii = dataset.nii_obj  
    original_affine = original_nii.affine  
    original_header = original_nii.header  
    
    # 确保输出目录存在  
    # os.makedirs("work/self2self3d/image/images0225", exist_ok=True)  
    # os.makedirs("work/self2self3d/model", exist_ok=True)  
    
    for itr in range(max_iters):  
        for batch_data in dataloader:  
            # batch_data的shape: (1, 1, H, W)  
            # 若该切片全0，跳过  
            if torch.all(batch_data == 0):  
                continue  

            batch_data = batch_data.to(device)  
            model.train()  

            # 生成brain_mask: 输入>0的地方为1，否则0  
            brain_mask = (batch_data > 0).float()  

            # 在H、W两个维度 pad到 (512, 512)  
            batch_data_padded, pad_info = pad_h_and_w(batch_data, desired_size=512)  

            # 生成drop mask  
            p_mtx = torch.rand_like(batch_data_padded)  
            drop_mask = (p_mtx > p).float() * 0.7  

            # 前向  
            img_input = batch_data_padded * drop_mask  
            output_padded = model(img_input, drop_mask)  

            # 去除两维pad  
            output_unpadded = unpad_h_and_w(output_padded, pad_info)  
            drop_mask_unpad = unpad_h_and_w(drop_mask, pad_info)  

            # 仅在 (1 - drop_mask_unpad) & brain_mask 处计算loss  
            combined_mask = (1 - drop_mask_unpad) * brain_mask  
            if torch.sum(combined_mask) < 1e-6:  
                continue  

            loss = torch.sum((output_unpadded - batch_data)**2 * combined_mask) / torch.sum(combined_mask)  

            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  

            # 如仅测试可去掉此break  
            break  

        if (itr + 1) % 100 == 0:  # 增加打印频率，便于监控进度  
            print(f"Iteration {itr+1}, loss = {loss.item()*100:.4f}")  

        # 每1000次迭代保存一次模型和预测结果  
        if (itr + 1) % 20000 == 0:  
            model.eval()  
            vol_shape = dataset.vol.shape  # (W, 1, H, D)  
            sum_preds = np.zeros((vol_shape[0], vol_shape[2], vol_shape[3]), dtype=np.float32)  

            with torch.no_grad():  
                for _ in tqdm(range(NPred), desc=f"Predicting for {file_id}"):  
                    pred_idx = 0  
                    for d in range(vol_shape[0]):  
                        slice_data = dataset[d].unsqueeze(0)  # (1, 1, H, W)  
                        # 跳过全0切片，但在结果中要占位  
                        if torch.all(slice_data == 0):  
                            pred_idx += 1  
                            continue  

                        slice_data = slice_data.to(device)  

                        # 测试时也生成brain_mask  
                        brain_mask_test = (slice_data > 0).float()  

                        # pad到 (512, 512)  
                        slice_data_padded, pad_info = pad_h_and_w(slice_data, desired_size=512)  

                        p_mtx = torch.rand_like(slice_data_padded)  
                        mask_test = (p_mtx > p).float() * 0.7  
                        out_test_padded = model(slice_data_padded * mask_test, mask_test)  

                        # 去除两维pad  
                        out_test_unpad = unpad_h_and_w(out_test_padded, pad_info)  

                        # 只保留brain_mask=1的区域  
                        out_test_unpad = out_test_unpad * brain_mask_test  

                        out_np = out_test_unpad.cpu().numpy()[0, 0]  # (H, W)  
                        sum_preds[pred_idx] += out_np  
                        pred_idx += 1  

            # 累加结果求平均  
            avg_preds = sum_preds / float(NPred)  

            # 恢复形状 (W, 1, H, D)  
            avg_preds_3d = np.expand_dims(avg_preds, axis=1)  

            # 再转回 (D, H, W)  
            avg_preds_3d = np.squeeze(avg_preds_3d, axis=1)  # (W, H, D)  
            avg_preds_3d = np.transpose(avg_preds_3d, (2, 1, 0))  # (D, H, W)  

            # 保持原头信息  
            out_nii = nib.Nifti1Image(avg_preds_3d, affine=original_affine, header=original_header)  
            nib.save(out_nii, f"image/images0307/{file_id}_output_{itr+1}.nii.gz")  

            # 保存模型  
            # torch.save(model.state_dict(), f"work/self2self3d/model/{file_id}_model_{itr+1}.pth")  
            
    print(f"=== 完成训练文件: {file_id} ===\n")  
    return model  

if __name__ == "__main__":  
    # 扫描目录找到所有满足条件的文件  
    dataset_path = "/data/birth/lmx/work/Class_projects/course5/dataset/Fetal_Brain_dataset/GMH_IVH_data/img"  
    pattern = re.compile(r'pa(\d+)\.nii\.gz')  
    
    file_list = []  
    
    for filename in os.listdir(dataset_path):  
        match = pattern.match(filename)  
        if match:  
            file_num = int(match.group(1))  
            if file_num >= 51:  # 只包含fb0391及之后的文件  
                file_path = os.path.join(dataset_path, filename)  
                file_id = f"pa{file_num:03d}"  
                file_list.append((file_path, file_id))  

    # 按文件编号排序  
    file_list.sort(key=lambda x: x[1])  
    
    print(f"找到 {len(file_list)} 个文件待处理:")  
    for _, file_id in file_list:  
        print(f"- {file_id}")  
    
    # 依次训练每个文件  
    for file_path, file_id in file_list:  
        # 为每个文件训练一个单独的Self2Self模型  
        model = train_self2self_for_file(file_path, file_id, max_iters=20010)  