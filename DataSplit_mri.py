import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import nibabel as nib
from monai.transforms import ScaleIntensity, NormalizeIntensity
from torchvision import transforms
from skimage.transform import resize
import random
import copy

class DataSplit_mri(nn.Module):
    def __init__(self, config, phase='train', do_transform=True):
        super(DataSplit_mri, self).__init__()

        if phase == 'train':
            self.data_csv = pd.read_csv(config.train_list)
        elif phase == 'valid':
            self.data_csv = pd.read_csv(config.valid_list)
        else:
            self.data_csv = pd.read_csv(config.test_list)

        self.t1_dir = config.t1_dir
        self.dti_dir = config.dti_dir

        self.do_transform = do_transform
        transform_list = []
        transform_list.append(transforms.ToTensor())
        #transform_list.append(ScaleIntensity(minv=0.0, maxv=1.0))
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        self.transform = transforms.Compose(transform_list)

        ## Subject & Gradient
        total_subs = list(self.data_csv.iloc[:,1])
        
        self.subs = []
        self.struct = []
        self.dti = []
        for i in range(len(self.data_csv)):
            sub = self.data_csv.iloc[i][1]
            
            t1 = np.load(self.t1_dir + '/' + sub + '.T1_256.npy')
            dti = np.load(self.dti_dir + '/' + sub + '.FA_256.npy')

            """
            t1 = np.array(nib.load(self.t1_dir + '/' + sub + '.brain.nii.gz').get_fdata())
            t1 = t1[:, :, int(t1.shape[0]/2)]

            dti = np.array(nib.load(self.dti_dir + '/' + sub + '.FA.nii.gz').get_fdata())
            dti = dti[:, :, int(dti.shape[0]/2)]
            """

            self.subs.append(sub)
            self.struct.append(t1)
            self.dti.append(dti)

        self.subs = np.array(self.subs)
        self.total_struct = np.array(self.struct)
        self.total_dti = np.array(self.dti)

    def __len__(self):
        assert len(self.total_struct) == len(self.total_dti)
        return len(self.total_struct)

    def __getitem__(self, index):
        sub = self.subs[index]
        t1 = self.total_struct[index]
        dti = self.total_dti[index]
    
        t1 /= np.max(t1)

        ## Transform
        if self.do_transform is not None:
            t1 = self.transform(t1).float() 
            dti = self.transform(dti).float() 
        # struct = torch.cat((t1, b0), dim=0) 

        return {"subject":sub, "real_A": t1, "real_B": dti}
