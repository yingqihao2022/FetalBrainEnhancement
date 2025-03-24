import numpy as np
import torch
import torch.nn as nn
from torch.nn import L1Loss
# from dataset import *
from torch.utils.data import DataLoader
# from unet_seg_sor import unet_seg
import os

import sys
sys.path.append('/data/birth/lmx/work/Class_projects/course5/work/vae_hj')
# from generative.networks.nets import AutoencoderKL, PatchDiscriminator

from autoencoderkl import AutoencoderKL
from adversarial_loss import PatchAdversarialLoss
from perceptual import PerceptualLoss
from vaedataset import UKBDataset
from tqdm import tqdm


# sys.path.append('/data/birth/yhj/fetal_SR/Seg_0727')
# from dice import DiceLoss


os.environ['CUDA_VISIBLE_DEVICES'] = '6'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
ckpt_dir = '/data/birth/lmx/work/Class_projects/course5/work/vae_hj/ckpt_single'
batch_size = 8


#input data
dataset_train = UKBDataset(setname='train')
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=24,drop_last=True,pin_memory=True)
num_data_train = len(dataset_train)
num_batch_train = np.ceil(num_data_train / batch_size)
dataset_val = UKBDataset(setname='val')
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=24, drop_last=True,pin_memory=True)
num_data_val = len(dataset_val)
num_batch_val = np.ceil(num_data_val / batch_size)

# define model
autoencoder= AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=(64, 128, 192),
    latent_channels=3,
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False,False,True),
)
autoencoder.to(device)

# discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
# discriminator.to(device)

# net= unet_seg()  
# net.to(device)
# state_dict = torch.load("MinValLoss.pth", map_location=device)
# net.load_state_dict(state_dict['net'])
# net.eval()

#loss function 
l1_loss = L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")
loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
loss_perceptual.to(device)
def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]
# adv_weight = 0.01
perceptual_weight = 0.001
kl_weight = 1e-6
# Loss_seg = DiceLoss(9)
MinValLoss=10000

#Train parameters
n_epochs = 300
st_epoch = 0
autoencoder_warm_up_n_epochs = 5

#optimizer
optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=0.0005)#1e-4
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer =optimizer_g,T_max=n_epochs)
#optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=0.00005)#1e-4
# optimizer_s = torch.optim.Adam(params=net.parameters(), lr=0.00005)

#Train
log=np.zeros([n_epochs,8])
for epoch in tqdm(range(st_epoch+1, n_epochs+1)):
    epoch_recon_loss_list = []
    # epoch_gen_loss_list = []
    # epoch_disc_loss_list = []
    # epoch_dice_loss_list = []

    autoencoder.train()
    # discriminator.train()
    # net.train()

    # if epoch%4==0:
    #     loader_train=loader_train_1
    # elif epoch%4==1:
    #     loader_train=loader_train_2
    # elif epoch%4==2:
    #     loader_train=loader_train_3
    # elif epoch%4==3:
    #     loader_train=loader_train_4


    #for step, batch in progress_bar:
    for batch, data in enumerate(loader_train, 1):
        images = data['high'].to(device)  # choose only one of Brats channels
        #age    = data['age'].to(device)
        # Generator part
        optimizer_g.zero_grad(set_to_none=True)
        # optimizer_s.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = autoencoder(images)
        # output_seg=net(reconstruction)

        kl_loss = KL_loss(z_mu, z_sigma)
        recons_loss = l1_loss(reconstruction.float(), images.float())
        p_loss = loss_perceptual(reconstruction.float(), images.float())
        # segloss,per_ch_score = Loss_seg(output_seg, data['seg'].to(device))

        loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss #+ segloss

        '''
        if epoch > autoencoder_warm_up_n_epochs:
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = loss_g + adv_weight * generator_loss
            epoch_gen_loss_list += [generator_loss.item()]
        '''

        # epoch_dice_loss_list += [segloss.item()]
        epoch_recon_loss_list += [recons_loss.item()]

        loss_g.backward()
        optimizer_g.step()
        # optimizer_s.step()

        '''
        if epoch > autoencoder_warm_up_n_epochs:
            # Discriminator part
            optimizer_d.zero_grad(set_to_none=True)
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            loss_d = adv_weight * discriminator_loss

            epoch_disc_loss_list += [discriminator_loss.item()]

            loss_d.backward()
            optimizer_d.step()
        '''
        print("EPOCH %04d / %04d | BATCH %04d / %04d | recons_loss %.4f " %  (epoch, n_epochs, batch, num_batch_train, np.mean(epoch_recon_loss_list)))
        # print("EPOCH %04d / %04d | BATCH %04d / %04d | recons_loss %.4f | gen_loss %.4f | disc_loss %.4f | dice_loss %.4f" %  (epoch, n_epochs, batch, num_batch_train, np.mean(epoch_recon_loss_list), np.mean(epoch_gen_loss_list), np.mean(epoch_disc_loss_list), np.mean(epoch_dice_loss_list)))
        #print("EPOCH %04d / %04d | BATCH %04d / %04d | recons_loss %.4f | gen_loss %.4f | disc_loss %.4f" %  (epoch, n_epochs, batch, num_batch_train, np.mean(epoch_recon_loss_list), np.mean(epoch_gen_loss_list), np.mean(epoch_disc_loss_list)))

    scheduler.step()
    log[epoch-1,0]=np.mean(epoch_recon_loss_list)
    # log[epoch-1,1]=np.mean(epoch_gen_loss_list)
    # log[epoch-1,2]=np.mean(epoch_disc_loss_list)
    # log[epoch-1,6]=np.mean(epoch_dice_loss_list)


    with torch.no_grad():
        autoencoder.eval()
        #discriminator.eval()
        epoch_recon_loss_list = []
        # epoch_gen_loss_list = []
        # epoch_disc_loss_list = []
        # epoch_dice_loss_list = []
        for batch, data in enumerate(loader_val, 1):

            images = data['high'].to(device) 
            reconstruction, z_mu, z_sigma = autoencoder(images)
            # output_seg=net(reconstruction)
     
            kl_loss = KL_loss(z_mu, z_sigma)
            recons_loss = l1_loss(reconstruction.float(), images.float())
            epoch_recon_loss_list += [recons_loss.item()]
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            # segloss,per_ch_score = Loss_seg(output_seg, data['seg'].to(device))

            # epoch_dice_loss_list += [segloss.item()]
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss #+ segloss
     
            '''
            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_g + adv_weight * generator_loss
                epoch_gen_loss_list += [generator_loss.item()]
    
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = adv_weight * discriminator_loss
     
                epoch_disc_loss_list += [discriminator_loss.item()]
            '''
            print("VAL EPOCH %04d / %04d | BATCH %04d / %04d | recons_loss %.4f " %  (epoch, n_epochs, batch, num_batch_val, np.mean(epoch_recon_loss_list)))
            # print("VAL EPOCH %04d / %04d | BATCH %04d / %04d | recons_loss %.4f | gen_loss %.4f | disc_loss %.4f | dice_loss %.4f" %  (epoch, n_epochs, batch, num_batch_val, np.mean(epoch_recon_loss_list), np.mean(epoch_gen_loss_list), np.mean(epoch_disc_loss_list), np.mean(epoch_dice_loss_list)))
     
        log[epoch-1,3]=np.mean(epoch_recon_loss_list)
        # log[epoch-1,4]=np.mean(epoch_gen_loss_list)
        # log[epoch-1,5]=np.mean(epoch_disc_loss_list)
        # log[epoch-1,7]=np.mean(epoch_dice_loss_list)
    np.save('log2.npy',log)

    torch.save({'net': autoencoder.state_dict(), 'optim' : optimizer_g.state_dict(), 'epoch':epoch, 'lr_schedule':scheduler.state_dict()},"%s/autoencoder_eachepoch.pth" % (ckpt_dir))
    #torch.save({'net': autoencoder.state_dict(), 'optim' : optimizer_g.state_dict(), 'epoch':epoch},"%s/autoencoder_eachepoch.pth" % (ckpt_dir))
    # torch.save({'net': net.state_dict(), 'optim' : optimizer_s.state_dict()},"%s/seg_eachepoch.pth" % (ckpt_dir))
    if np.mean(epoch_recon_loss_list)<MinValLoss:
        MinValLoss=np.mean(epoch_recon_loss_list)
        torch.save({'net': autoencoder.state_dict(), 'optim' : optimizer_g.state_dict(), 'epoch':epoch, 'lr_schedule':scheduler.state_dict()},"%s/autoencoder_minloss.pth" % (ckpt_dir))
        #torch.save({'net': autoencoder.state_dict(), 'optim' : optimizer_g.state_dict(), 'epoch':epoch},"%s/autoencoder_minloss.pth" % (ckpt_dir))
        # torch.save({'net': net.state_dict(), 'optim' : optimizer_s.state_dict()},"%s/seg_minloss.pth" % (ckpt_dir))

# del discriminator
# del loss_perceptual
# torch.cuda.empty_cache()