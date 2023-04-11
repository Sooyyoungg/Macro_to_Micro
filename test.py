import random
import torch
import pandas as pd
import tensorboardX
import cv2
import imageio
import numpy as np

from sklearn.metrics import mean_squared_error
import torchsummary
import time
from datetime import datetime
from pytz import timezone
import imageio
from matplotlib import pyplot as plt
from PIL import Image

from Config import Config
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
            sync_file = 'file://%s/pytorch_sync.%s.%s' % (                                                          sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
        else:
            if 'SYNC_CODE' in os.environ:
                sync_file = 'file://%s/pytorch_sync.%s.%s' % (
                sync_file_dir, os.environ['SYNC_CODE'], os.environ['SYNC_CODE'])
            else:
                sync_file = 'file://%s/pytorch_sync.%s.%s' % (
                sync_file_dir, 12345, 12345) 
        return sync_file

def mkoutput_dir(config):
    if not os.path.exists(config.img_dir+'/Test'):
        os.makedirs(config.img_dir+'/Test')

def main():
    config = Config()
    
    ## DDP
    # sbatch script에서 WORLD_SIZE를 지정해준 경우 (노드 당 gpu * 노드의 수)
    if "WORLD_SIZE" in os.environ:  # for torchrun
        config.world_size = int(os.environ["WORLD_SIZE"])
    # 혹은 슬럼에서 자동으로 ntasks per node * nodes 로 구해줌
    elif 'SLURM_NTASKS' in os.environ:
        config.world_size = int(os.environ['SLURM_NTASKS'])
    else:
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
        print('distributed gpus:', config.gpu)
        sync_file = _get_sync_file()
        dist.init_process_group(backend=config.dist_backend, init_method=sync_file,
                            world_size=config.world_size, rank=config.rank)
    else:
        config.rank = 0
        config.gpu = 0

    # suppress printing if not on master gpu
    if config.rank!=0:
        def print_pass(*config):
            pass
        builtins.print = print_pass

    #device = torch.device('cuda:{}'.format(config.gpu_ids[0])) if config.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
    print('cuda:', config.gpu)

    ## Data Loader
    test_data = DataSplit_mri(config=config, phase='test')
    data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=16, pin_memory=False)
    print("Test: ", test_data.__len__(), "images: ", len(data_loader_test), "x", 1,"(batch size) =", test_data.__len__())

    if (not config.distributed) or config.rank == 0 :
        mkoutput_dir(config)

    ## Model load
    model = OCCAY(config)
    
    # load saved model
    model, model.optimizer_E, model.optimizer_G, model.optimizer_D, model.optimizer_PD, epoch_start = model_load(save_type='normal', ckpt_dir=config.ckpt_dir, model=model,
                           optim_E=model.optimizer_E,
                           optim_G=model.optimizer_G,
                           optim_D=model.optimizer_D,
                           optim_PD=model.optimizer_PD)
    print(epoch_start+1, "th model load")

    model.to(model.device)

    ## Start Testing
    print("Start Testing!!")
    start_time = datetime.now(timezone('Asia/Seoul'))
    print ('Test start time: %s month, %s day, %s h, %s m and %s s.' % (start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))
    
    with torch.no_grad():
        for i, data in enumerate(data_loader_test):
            test_dict = model.test(data)

            sub = data['subject']
            real_A = data['real_A']
            fake_B = test_dict['fake_AtoB']
            real_B = data['real_B']

            # post-processing: [-1, 1] -> [0, 1]
            real_A = np.clip(((real_A.detach().cpu().numpy() + 1) / 2), 0, 1)
            real_B = np.clip(((real_B.detach().cpu().numpy() + 1) / 2), 0, 1)
            fake_B = np.clip(((fake_B.detach().cpu().numpy() + 1) / 2), 0, 1)

            ## Generated Image Save
            print(i,"th image save")
            real_A_g = real_A[0][0]*255.0
            real_B_g = real_B[0][0]*255.0
            fake_B_g = fake_B[0][0]*255.0

            A_image = Image.fromarray(real_A_g.astype(np.uint8), 'L')
            B_image = Image.fromarray(real_B_g.astype(np.uint8), 'L')
            f_image = Image.fromarray(fake_B_g.astype(np.uint8), 'L')

            # save
            print(config.img_dir+'/Test/'+str(i+1)+'_'+sub[0]+'_A.png')
            A_image.save(config.img_dir+'/Test/'+str(i+1)+'_'+sub[0]+'_A.png')
            f_image.save(config.img_dir+'/Test/'+str(i+1)+'_'+sub[0]+'_fake.png')
            B_image.save(config.img_dir+'/Test/'+str(i+1)+'_'+sub[0]+'_B.png')
                 
            # print("Loss_G: %.5f | Loss_D: %.5f"%(test_dict['G_loss'], test_dict['D_loss']))

    end_time = datetime.now()
    print ('Test start time: %s month, %s day, %s hours, %s minutes and %s seconds.' % (start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second))
    print ('Test finish time: %s month, %s day, %s hours, %s minutes and %s seconds.' % (end_time.month, end_time.day, end_time.hour, end_time.minute, end_time.second))

if __name__ == '__main__':
    main()
