# This is a sample Python script.
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from utils.utils import data_prefetcher_two, cal_fam, setup_seed, calRes 
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import argparse
import random
import time
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched
#from data import mydataset,traintransform
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from utils_1 import writeImage
import torch.optim as optim
from timm.models.vision_transformer import PatchEmbed, Block
import torch.nn as nn
import timm
import torch.backends.cudnn as cudnn
import os
from PIL import Image
from torch.utils import data
from torch.utils.data import  ConcatDataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import os
from PIL import Image
from torch.utils import data
from torch.utils.data import  ConcatDataset
import cv2
from sklearn.metrics import roc_auc_score
import torch
from utils.utils import data_prefetcher_two, cal_fam, setup_seed, calRes
from xpection import xception
import utils.datasets_profiles as dp
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import argparse
import random
import time

import torch
import torch.nn.functional as F

import torchvision
from torchvision import models






traintransform = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor()])
testtransform = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor()])
masktransform = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor()])

            
###### dataloader: each viedos 50 frames
class mydataset(data.Dataset):
    def __init__(self, root,transforms=None,train=True):
        self.transforms = transforms
        self.imgs = []
        dir1 = root
        name = 'fake'
        self.train = train
        if(self.train):
            name = 'real'
        dir1list = os.listdir(dir1)
        for i in range(len(dir1list)):
            if(dir1list[i].find('.')==-1):
                dir2 = dir1 + dir1list[i] + '/'
                dir2list = os.listdir(dir2)
                aaa = 0
                for j in range(len(dir2list)):
                    if(dir2list[j].find('.') !=-1):
                    #print(dir2list[j])
                        if(aaa>50):
                            break
                        if( int(dir2list[j][0:4])%4 ==0 ):
                            dir3 = dir2 + dir2list[j]
                            self.imgs.append(dir3)
                            aaa = aaa + 1
                        
    def __getitem__(self,index):
        label = [0.0 ,1.0]
        if(self.train):
            label = [1.0 ,0.0]
        label = torch.tensor(label)
        img_pth = self.imgs[index]
        data = self.transforms(Image.open(img_pth))
        return data,label
        
    def __len__(self):
        return len(self.imgs)





############################################  frequecy remove  numpyt vision
########## input: imgs - (input images), mask_rate - (high-frequency mask rate)
########## output: high frequency (masked) images, low frequency images
######### separate low frequency and high frequency (masked)
###########################################################################

def get_hai(imgs,mask_rate):
    h = []
    l = []
    for i in range(imgs.shape[0]):
        h1 = []
        l1 = []
        for j in range(3):
            f = np.fft.fft2(imgs[i][j].reshape(224,224))
            fshift = np.fft.fftshift(f)
            rows,cols = fshift.shape
            mid_x,mid_y = int((rows)/2),(int((cols)/2))
            mask1 = np.ones((rows,cols),dtype=np.uint8)
            ########   you can change d in here 
            mask1[mid_x-5:mid_x+5,mid_y-5:mid_y+5] = 0
            fshift1 = mask1*fshift
            isshift1 = np.fft.ifftshift(fshift1)
            mask2 = np.zeros((rows,cols),dtype=np.uint8)
            ########   you can change d in here 
            mask2[mid_x-5:mid_x+5,mid_y-5:mid_y+5] = 1
            fshift2 = mask2*fshift
            isshift2 = np.fft.ifftshift(fshift2)
            high = np.fft.ifft2(isshift1)
            low = np.fft.ifft2(isshift2)
            img_high = np.abs(high)
            img_low = np.array(imgs[i][j])-np.array(img_high)
            h1.append(img_high)
            l1.append(img_low)
        #print(len(h1))
        h.append(h1)
        l.append(l1)
    h = np.array(h).reshape(-1,3,224,224).astype('float32')
    l = np.array(l)
    mask = [random.random() for _ in range(224*224)]
    mask = np.array(mask)
    mask = np.int64(mask>mask_rate)
    return h*mask.reshape(224,224).astype('float32'),l.astype('float32')


############################### eval function 
################   input:  model - (forgery detector), dtloader - (test dataloader), epoch - (now training epoch)
################   output: AUC 
###############    eval the model in each epoch and save test images

def Eval(model, dtloader,epoch):
    model.eval()
    sumloss = 0.
    y_true_all = None
    y_pred_all = None
    print(len(dtloader))
    with torch.no_grad():
        for (j, batch) in enumerate(dtloader):
            x, y_true = batch
            y_pred = model(x.cuda())

            imgs = [x.cuda(),x.cuda()]
            y_predx = y_true.cuda()
            ####### sigmoid function 
            for i in range(len(y_true)):
                y_pred[i][0] = 1/(1+torch.exp(-1*y_pred[i][0]))
                y_pred[i][1] = 1/(1+torch.exp(-1*y_pred[i][1]))
                sum1 =  y_pred[i][0]+y_pred[i][1]
                y_pred[i][0] = y_pred[i][0]/sum1
                y_pred[i][1] = y_pred[i][1]/sum1                      
                if(y_pred[i][0]>=y_pred[i][1]):
                    y_predx[i] =torch.tensor([0,1])
                else:
                    y_predx[i] =torch.tensor([1,0])

            if y_true_all is None:
                y_true_all = y_true
                y_pred_all = y_predx
                auc = y_pred
            else:
                y_true_all = torch.cat((y_true_all, y_true))
                y_pred_all = torch.cat((y_pred_all, y_predx))
                auc = torch.cat((auc,y_pred))


        a =  y_true_all.detach()
        b =  y_pred_all.detach()
        d = np.array(y_true_all.cpu())
        e = np.array(auc.cpu())
        print(d[:,0].reshape(-1))
        print(e[:,0].reshape(-1))
        print('AUC',roc_auc_score(d[:,0],e[:,0]))
        #print(a,b)
        sumcnt = 0
        a=a.cuda()
        for i in range(len(a)):
            if(a[i][0]==b[i][1]):
                sumcnt += 1
        print(sumcnt/len(a))
    return roc_auc_score(d[:,0],e[:,0])









##### define model and lossfunc
model = xception(num_classes=2, pretrained=False).cuda()
seed = 0
setup_seed(seed)
lossfunc = torch.nn.CrossEntropyLoss()

#### define test dataloader
dir3 = '/Harddisk/Datasets/1. [2019 Database & arXiv] Celeb-DF_A new dataset/fake_test/images/'
testset1 = mydataset(dir3,traintransform,train=False)
dir4 =  '/Harddisk/Datasets/1. [2019 Database & arXiv] Celeb-DF_A new dataset/real_test/images/'
testset2 = mydataset(dir4,traintransform,train=True)
testset =  ConcatDataset([testset1,testset2])
testDataloader = torch.utils.data.DataLoader(testset, batch_size=512,shuffle=False, num_workers=int(1), drop_last=False)

del testset1
del testset2




batchind = 0
e = 0
sumcnt = 0
sumloss = 0.0
iter = 1
ans = []
loss1 = 0
loss2 = 0
#accum_iter = args.accum_iter


###### loader the model
import collections
from collections import OrderedDict
predtrain = torch.load('/Harddisk/paper2/model/parameter/mask_ratio/0.1/checkpoint-5.pth')
us_dict = []
new_state_dict = OrderedDict()
for k,v in predtrain['model'].items():
    name = k.replace('module.','')
    print(name)
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)




###### define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
print(optimizer)


##### test the data_loader
auc = Eval(model,testDataloader,0)



