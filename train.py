import numpy as np
import random
import torch
import pandas as pd
import tensorboardX
import cv2
from sklearn.metrics import mean_squared_error
import torchsummary
import time
from datetime import datetime
from pytz import timezone
import imageio
from matplotlib import pyplot as plt
from PIL import Image

from Config import Config
from DataSplit import DataSplit
from DataSplit_mri import DataSplit_mri
from model import OCCAY
from blocks import model_save, model_load
import networks

#DDP
import os
import builtins
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

def _get_sync_file():    
        """Logic for naming sync file using slurm env variables"""
        if 'SCRATCH' in os.environ:
            sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH'] # Perlmutter
        else:
            raise Exception('there is no env variable SCRATCH. Please check sync_file dir')
        os.makedirs(sync_file_dir, exist_ok=True)

        #temporally add two lines below for torchrun
        if ('SLURM_JOB_ID' in os.environ) and ('SLURM_STEP_ID' in os.environ) :
            sync_file = 'file://%s/pytorch_sync.%s.%s' % (
            sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
        else:
            if 'SYNC_CODE' in os.environ:
                sync_file = 'file://%s/pytorch_sync.%s.%s' % (
                sync_file_dir, os.environ['SYNC_CODE'], os.environ['SYNC_CODE'])
            else:
                sync_file = 'file://%s/pytorch_sync.%s.%s' % (
                sync_file_dir, 12345, 12345)
        return sync_file

def mkoutput_dir(config):
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)
    if not os.path.exists(config.img_dir):
        os.makedirs(config.img_dir)
    if not os.path.exists(config.img_dir+'/Train'):
        os.makedirs(config.img_dir+'/Train')
        os.makedirs(config.img_dir+'/Validation')

def load_pretrained(model):
    model_dir = './oct_resnet50_ours.pth'
    model_state = torch.load(model_dir, map_location='cpu')
    model.netOct.load_state_dict(model_state, strict=False)
    return model

def main():
    config = Config()
    
    ## DDP
    # sbatch script에서 WORLD_SIZE를 지정해준 경우 (노드 당 gpu * 노드의 수)
    if "WORLD_SIZE" in os.environ:  # for torchrun
        config.world_size = int(os.environ["WORLD_SIZE"])
    # 혹은 슬럼에서 자동으로 ntasks per node * nodes 로 구해줌
    elif 'SLURM_NTASKS' in os.environ:
        config.world_size = int(os.environ['SLURM_NTASKS'])
    # SLURM 없는 경우
    else:
        config.world_size = torch.cuda.device_count()
        pass # torch.distributed.launch
    
    config.distributed = config.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if config.distributed:
        if config.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'RANK' in os.environ: # for torchrun
            config.rank = int(os.environ['RANK'])
            config.gpu = int(os.environ["LOCAL_RANK"])
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            config.rank = int(os.environ['SLURM_PROCID'])
            config.gpu = config.rank % torch.cuda.device_count()
        else:
            #config.gpu = [0, 1, 2, 3]
            config.gpu = [0, 1]
        print('distributed gpus:', config.gpu, " / rank:", config.rank)
        sync_file = _get_sync_file()
        dist.init_process_group(backend=config.dist_backend, init_method=sync_file,
                            world_size=config.world_size, rank=config.rank)
        #dist.init_process_group(backend=config.dist_backend, world_size=config.world_size)
    else:
        config.rank = 0
        config.gpu = 0
    
    # suppress printing if not on master gpu
    if config.rank!=0:
        def print_pass(*config):
            pass
        builtins.print = print_pass
    
    print('cuda:', config.gpu)

    ## Data Loader
    #train_list = pd.read_csv(config.train_list, header=None)
    #valid_list = pd.read_csv(config.valid_list, header=None)

    if config.data == 'mri':
        train_data = DataSplit_mri(config=config, phase='train')
        valid_data = DataSplit_mri(config=config, phase='valid')
    else:
        train_data = DataSplit(config=config, phase='train')
        valid_data = DataSplit(config=config, phase='valid')

    if config.distributed:
        train_sampler = DistributedSampler(train_data , shuffle=True)
        #valid_sampler = DistributedSampler(valid_data, shuffle=True) 
    else:
        train_sampler = RandomSampler(train_data)
        #valid_sampler = RandomSampler(valid_data)
       
    data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size,  num_workers=4, pin_memory=False, sampler=train_sampler)
    data_loader_valid = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, num_workers=4, pin_memory=False, sampler=None)
    
    print("Train: ", train_data.__len__(), "images: ", len(data_loader_train), "x", config.batch_size,"(batch size) =", train_data.__len__())
    print("Valid: ", valid_data.__len__(), "images: ", len(data_loader_valid), "x", config.batch_size,"(batch size) =", valid_data.__len__())

    # make log, ckpt, Generated_images output directory
    if (not config.distributed) or config.rank == 0 :
        mkoutput_dir(config)

    ## Model load
    model = OCCAY(config)
    
    #torchsummary.summary(model, input_size=[(1, 140, 140), (1, 140, 140)], device='cpu')
    
    # load saved model
    if config.train_continue == 'on':
        model, model.optimizer_E, model.optimizer_G, model.optimizer_D, epoch_start = model_load(save_type='normal', ckpt_dir=config.ckpt_dir, model=model, 
                           optim_E=model.optimizer_E,
                           optim_G=model.optimizer_G, 
                           optim_D=model.optimizer_D,
                           optim_PD=model.optimizer_PD)
        print(epoch_start+1, "th model load")
    else:
        epoch_start = -1

    #model = load_pretrained(model)
    model.to(model.device)

    train_writer = tensorboardX.SummaryWriter(config.log_dir)
    
    ## Start Training
    print("Start Training!!")
    start_time = datetime.now(timezone('Asia/Seoul'))
    print ('Train start time: %s month, %s day, %s h, %s m and %s s.' % (start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))
   
    tot_itr = 0
    min_v_G_loss = 100
    min_v_D_loss = 100
    for epoch in range(epoch_start+1, config.n_epoch):
        epoch_start = datetime.now(timezone('Asia/Seoul'))
       
        # 1/3 지점에서 high frequency weight = 0.7로 고정되도록
        config.freq_weight = 1.43 * (epoch + 1) / config.n_epoch
        print("freq_weight of ", epoch,"th epoch: ", config.freq_weight)

        if config.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            #data_loader_valid.sampler.set_epoch(epoch)
        
        for i, data in enumerate(data_loader_train):
            tot_itr += 1
            train_dict = model.train(tot_itr, data)

            sub = data['subject']
            real_A = data['real_A']
            real_B = data['real_B']
            fake_B = train_dict['fake_AtoB']    # [batch, 1, 128, 128]
            trs_low = train_dict['fake_AtoB_low']
            trs_high = train_dict['fake_AtoB_high']

            # post-processing: [-1, 1] -> [0, 1]
            real_A = (real_A.detach().cpu().numpy() + 1) / 2
            real_B = (real_B.detach().cpu().numpy() + 1) / 2
            fake_B = (fake_B.detach().cpu().numpy() + 1) / 2
            trs_low = (trs_low.detach().cpu().numpy() + 1) / 2
            trs_high = (trs_high.detach().cpu().numpy() + 1) / 2
           
            ## Generated Image Save
            if i % 20 == 0:
                #r = random.randint(0, config.batch_size-1)
                #image rescaling & save
                if config.input_nc == 1:
                    real_A_g = np.reshape(real_A[0],(config.load_size, config.load_size))
                    real_B_g = np.reshape(real_B[0],(config.load_size, config.load_size))
                    fake_B_g = np.reshape(fake_B[0],(config.load_size, config.load_size))

                    A_image = Image.fromarray(np.clip(real_A_g, 0, 1).astype(np.uint8), 'L')
                    B_image = Image.fromarray(np.clip(real_B_g, 0, 1).astype(np.uint8), 'L')
                    f_image = Image.fromarray(np.clip(fake_B_g, 0, 1).astype(np.uint8), 'L')
                else:
                    A_image = Image.fromarray(np.clip(real_A[0] * 255.0, 0, 255).transpose(1,2,0).astype(np.uint8))
                    B_image = Image.fromarray(np.clip(real_B[0] * 255.0, 0, 255).transpose(1,2,0).astype(np.uint8))
                    f_image = Image.fromarray(np.clip(fake_B[0] * 255.0, 0, 255).transpose(1,2,0).astype(np.uint8))    
                # save
                A_image.save('{}/Train/{}_{}_A.png'.format(config.img_dir, epoch+1, i+1))
                f_image.save('{}/Train/{}_{}_fake.png'.format(config.img_dir, epoch+1, i+1))
                B_image.save('{}/Train/{}_{}_B.png'.format(config.img_dir, epoch+1, i+1))

            ## Tensorboard ##
            if (not config.distributed) or config.rank == 0 :

                # tensorboard - loss
                #train_writer.add_scalar('G_PD_sty', train_dict['G_PD_sty'], tot_itr)
                train_writer.add_scalar('Loss_G_Perceptual', train_dict['G_percept'], tot_itr)
                train_writer.add_scalar('Loss_G_PatchGAN', train_dict['G_PD'], tot_itr)
                train_writer.add_scalar('Loss_G_Trs', train_dict['G_trs'], tot_itr)
                train_writer.add_scalar('Loss_G_pix', train_dict['G_pix'], tot_itr)
                train_writer.add_scalar('Loss_G_img', train_dict['G_img'], tot_itr)
                train_writer.add_scalar('Loss_G_GAN', train_dict['G_GAN'], tot_itr)
                train_writer.add_scalar('Loss_G', train_dict['G_loss'], tot_itr)
                train_writer.add_scalar('Loss_D_PatchGAN', train_dict['D_PD'], tot_itr)
                train_writer.add_scalar('Loss_D_GAN', train_dict['D_GAN'], tot_itr)
                train_writer.add_scalar('Loss_D', train_dict['D_loss'], tot_itr)

                # tensorboard - images
                fn_tonumpy = lambda x: x.transpose(0, 2, 3, 1)
                train_writer.add_image('Content_Image_A', np.clip(fn_tonumpy(real_A), 0, 1), tot_itr, dataformats='NHWC')
                train_writer.add_image('Style_Image_B', np.clip(fn_tonumpy(real_B), 0, 1), tot_itr, dataformats='NHWC')
                train_writer.add_image('Generated_Image_AtoB', np.clip(fn_tonumpy(fake_B), 0, 1), tot_itr, dataformats='NHWC')
                train_writer.add_image('Translation_low', np.clip(fn_tonumpy(trs_low), 0, 1), tot_itr, dataformats='NHWC')
                train_writer.add_image('Translation_high', np.clip(fn_tonumpy(trs_high), 0, 1), tot_itr, dataformats='NHWC')


            # tensorboard - data visualization for 3D
            # features = fake_B.view(-1, fake_B.shape[-1]**2)
            # train_writer.add_embedding(features, label_img=fake_B)
            # train_writer.add_embedding(features, metadata=label, label_img=fake_B)
            
                print("Epoch: %d/%d | itr: %d/%d | tot_itrs: %d | Loss_G: %.5f | Loss_D: %.5f"%(epoch+1, config.n_epoch, i+1, len(data_loader_train), tot_itr, train_dict['G_loss'], train_dict['D_loss']))

        networks.update_learning_rate(model.E_scheduler, model.optimizer_E)
        networks.update_learning_rate(model.G_scheduler, model.optimizer_G)
        networks.update_learning_rate(model.D_scheduler, model.optimizer_D)
        networks.update_learning_rate(model.PD_scheduler, model.optimizer_PD)
       
        if (not config.distributed) or config.rank == 0 :
            ## Model Save
            # save every epoch
            model_save(save_type='normal', ckpt_dir=config.ckpt_dir, model=model, optim_E=model.optimizer_E, optim_G=model.optimizer_G, optim_D=model.optimizer_D, optim_PD=model.optimizer_PD, epoch=epoch)
            print(epoch+1, "th model save")

            ## Validation
            with torch.no_grad():
                valid_G_loss = 0
                valid_D_loss = 0
                
                for v, v_data in enumerate(data_loader_valid):
                    val_dict = model.val(v_data)
                    valid_G_loss += val_dict['G_loss']
                    valid_D_loss += val_dict['D_loss']
                    
                    v_real_A = val_dict['real_A'].detach().cpu().numpy()
                    v_fake_B = val_dict['fake_AtoB'].detach().cpu().numpy()
                    v_real_B = val_dict['real_B'].detach().cpu().numpy()

                    v_real_A = np.clip(((v_real_A + 1) / 2), 0, 1)
                    v_real_B = np.clip(((v_real_B + 1) / 2), 0, 1)
                    v_fake_B = np.clip(((v_fake_B + 1) / 2), 0, 1)

                    # save image
                    ## post-processing for image saving
                    v_real_A_g = np.reshape(v_real_A[0],(config.load_size, config.load_size))
                    v_real_B_g = np.reshape(v_real_B[0],(config.load_size, config.load_size))
                    v_fake_B_g = np.reshape(v_fake_B[0],(config.load_size, config.load_size))

                    v_A_image = Image.fromarray(v_real_A_g.astype(np.uint8), 'L')
                    v_f_image = Image.fromarray(v_fake_B_g.astype(np.uint8), 'L')
                    v_B_image = Image.fromarray(v_real_B_g.astype(np.uint8), 'L')

                    # save
                    v_A_image.save('{}/Validation/{}_A.png'.format(config.img_dir, epoch+1))
                    v_f_image.save('{}/Validation/{}_fake.png'.format(config.img_dir, epoch+1))
                    v_B_image.save('{}/Validation/{}_B.png'.format(config.img_dir, epoch+1))

                v_G_avg_loss = float(valid_G_loss / (v+1))
                v_D_avg_loss = float(valid_D_loss / (v+1))

                train_writer.add_scalar('Val_Loss_G', v_G_avg_loss, epoch)
                train_writer.add_scalar('Val_Loss_D', v_D_avg_loss, epoch)
                print("===> Validation <=== Epoch: %d/%d | Loss_G: %.5f | Loss_D: %.5f"%(epoch+1, config.n_epoch, v_G_avg_loss, v_D_avg_loss))

            # save best performance model
            if v_G_avg_loss < min_v_G_loss and v_D_avg_loss < min_v_G_loss:
                model_save(save_type='best', ckpt_dir=config.ckpt_dir, model=model, optim_E=model.optimizer_E, optim_G=model.optimizer_G, optim_D=model.optimizer_D, optim_PD=model.optimizer_PD, epoch=epoch)
                print("best model save")
        # else:
        #     dist.barrier()

        end_time = datetime.now(timezone('Asia/Seoul'))
        print ('Epoch start time: %s month, %s day, %s h, %s m and %s s.' % (epoch_start.month, epoch_start.day, epoch_start.hour, epoch_start.minute, epoch_start.second))
        print ('Epoch finish time: %s month, %s day, %s h, %s m and %s s.' % (end_time.month, end_time.day, end_time.hour, end_time.minute, end_time.second))

        
if __name__ == '__main__':
    main()
