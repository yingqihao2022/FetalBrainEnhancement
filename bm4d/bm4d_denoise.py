import numpy as np  
import nibabel as nib  
from scipy import fftpack  
import bm4d  
import argparse  
from tqdm import tqdm  
import os  

def process_bm4d(input_path, output_path):  
    
    data_affine = nib.load(input_path).affine  
    data = nib.load(input_path).get_fdata()  

    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))  
    nx, ny, nz = data_norm.shape  
    psd_acc = np.zeros((nx//2, ny//2))   
    for z in range(nz):  
        slice_2d = data_norm[:, :, z]  
        f2d = fftpack.fft2(slice_2d)  
        f2d_shifted = fftpack.fftshift(f2d)  
        psd_2d = np.abs(f2d_shifted)**2  
        psd_acc += psd_2d[:nx//2, :ny//2]  

    psd_avg = psd_acc / nz  
    y_hat = bm4d.bm4d(data_norm, psd_avg)  
    nib.Nifti1Image(y_hat, data_affine).to_filename(output_path)  
    print(f'Saved denoised file to: {output_path}')  

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Process BM4D for fetal brain dataset.')  
    parser.add_argument('input_path', type=str)  
    parser.add_argument('output_path', type=str)  

    args = parser.parse_args()  
    process_bm4d(args.input_path, args.output_path) 
