# This is a sample Python script.
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from utils.utils import data_prefetcher_two, setup_seed, calRes 
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
from utils.utils import data_prefetcher_two, setup_seed, calRes
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
#from attack import attack
import torchvision
from torchvision import models
from PIL import ImageEnhance

from torch.autograd import Variable
import torch.fft
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from grad_cam import *




traintransform = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),])
testtransform = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor(),])
masktransform = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor()])



class mydataset(data.Dataset):
    def __init__(self, root,transforms=None,train=True,train_list=[]):
        self.transforms = transforms
        self.imgs = []
        dir1 = root
        self.train = train
        dir1list = os.listdir(dir1)
        for i in range(len(dir1list)):
            if(dir1list[i] in train_list):
                dir2 = dir1 +'/'+dir1list[i] + '/'
                #print(dir2)
                #exit(0)
                dir2list = os.listdir(dir2)
                for j in range(len(dir2list)):
                    if(dir2list[j].find('cfg')!=-1):
                        continue
                    if( int(dir2list[j][0:4])%1 ==0 ):
                         dir3 = dir2 + dir2list[j]
                         #print(dir3)
                         #exit(0)
                         self.imgs.append(dir3)
                        
    def __getitem__(self,index):
        label = [0.0 ,1.0]
        if(self.train):
            label = [1.0,0.0]
        label = torch.tensor(label)
        img_pth = self.imgs[index]
        data = self.transforms(Image.open(img_pth))
        return data,label
        
    def __len__(self):
        return len(self.imgs)

class mydataset1(data.Dataset):
    def __init__(self, root,transforms=None,train=True,train_list=[]):
        self.transforms = transforms
        self.imgs = []
        dir1 = root
        self.train = train
        dir1list = os.listdir(dir1)
        for i in range(len(dir1list)):
            if(dir1list[i] in train_list):
                dir2 = dir1 +'/'+dir1list[i] + '/'
                #print(dir2)
                #exit(0)
                dir2list = os.listdir(dir2)
                for j in range(len(dir2list)):
                    if(dir2list[j].find('cfg')!=-1):
                        continue
                    if( int(dir2list[j][0:4])%4 ==0 ):
                         dir3 = dir2 + dir2list[j]
                         #print(dir3)
                         #exit(0)
                         self.imgs.append(dir3)
                        
    def __getitem__(self,index):
        label = [0.0 ,1.0]
        if(self.train):
            label = [1.0,0.0]
        label = torch.tensor(label)
        img_pth = self.imgs[index]
        data = self.transforms(Image.open(img_pth))
        return data,label
        
    def __len__(self):
        return len(self.imgs)






############ define train and test loader
import json
with open('/Harddisk/Datasets/1. [2019 Database & ICCV] FaceForensics++/1. [2019 ICCV] FaceForensics++- Learning to Detect Manipulated Facial Images/FaceForensics-master/dataset/splits/train.json') as f:
    train_data = json.load(f)

train_list1 = []
train_list2 = []
test_list1 = []
test_list2 = []
for i in range(len(train_data)):
    train_list1.append(train_data[i][0]+'_'+train_data[i][1])
    #train_list1.append(train_data[i][1]+'_'+train_data[i][0])
    train_list2.append(train_data[i][0])
    #train_list2.append(train_data[i][1])
print(train_list1)
print( '071_054' in train_list1)
print( '645_645' in train_list1)

with open('/Harddisk/Datasets/1. [2019 Database & ICCV] FaceForensics++/1. [2019 ICCV] FaceForensics++- Learning to Detect Manipulated Facial Images/FaceForensics-master/dataset/splits/test.json') as f:
    test_data = json.load(f)
for i in range(len(test_data)):
    test_list1.append(test_data[i][0]+'_'+test_data[i][1])
    #test_list1.append(test_data[i][1]+'_'+test_data[i][0])
    test_list2.append(test_data[i][0])
    #test_list2.append(test_data[i][1])

print('len',len(test_list1))
print('len',len(test_list2))
print( '071_054' in test_list1)
print( '645_645' in test_list1)




#exit(0)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--max_batch', default=500000, type=int,help='try')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=112, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')


    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda:0',help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='/home/gpu/Desktop/chentao/mae_pretrain_vit_base.pth',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser





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



############################################  frequecy remove  torch vision
########## input: imgs - (input images), mask_rate - (high-frequency mask rate)
########## output: high frequency (masked) images, low frequency images
######### separate low frequency and high frequency (masked)
###########################################################################

def _get_(imgs,mask_rate):
    img_l = None
    for j in range(imgs.shape[0]):
        img = None
        for i in range(3):
            fft_img = torch.fft.fft2(imgs[j,i])
            fft_shift_img = torch.fft.fftshift(fft_img)
            filter_rate = 0.9
            h, w = fft_shift_img.shape[:2] # height and width
            cy, cx = int(h/2), int(w/2) # centerness
            rh, rw = int(filter_rate * cy), int(filter_rate * cx) # filter_size
            rh = 5
            rw = 5
            mask = torch.ones_like(fft_shift_img)
            mask[cy-5:cy+5, cx-5:cx+5] = 0
            ifft_shift_img = torch.fft.ifftshift(mask*fft_shift_img)
            ifft_img = torch.fft.ifft2(ifft_shift_img).reshape(1,224,224)
            if(img==None):
                img = ifft_img
            else:
                img = torch.concat([img,ifft_img],dim=0)
        img = img.reshape(1,3,224,224)
        if(img_l==None):
            img_l = img
        else:
            img_l = torch.concat([img_l,img],dim=0)
       
    img_h = torch.abs(img_l.real)
    img_l = imgs - img_h
    mask = [random.random() for _ in range(224*224)]
    mask = np.array(mask)
    mask = torch.tensor(np.int64(mask>mask_rate))
    return img_h*mask.reshape(224,224),img_l



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
    
    
    

args = get_args_parser()

print(args)


batch_size = 64



##### define model and lossfunc
model = xception(num_classes=2, pretrained=False).cuda()
lossfunc = torch.nn.CrossEntropyLoss(reduction='none')




setup_seed(0)


#### define test and train dataloader
dir1 =  '/Harddisk/Datasets/1. [2019 Database & ICCV] FaceForensics++/1. FaceForensics++/manipulated_sequences/FaceSwap/c23/images'
trainset1 = mydataset1(dir1,traintransform,train=False,train_list=train_list1)
dir2 =  '/Harddisk/Datasets/1. [2019 Database & ICCV] FaceForensics++/1. FaceForensics++/manipulated_sequences/NeuralTextures/c23/images'
trainset2 = mydataset1(dir2,traintransform,train=False,train_list=train_list1)
dir3 =  '/Harddisk/Datasets/1. [2019 Database & ICCV] FaceForensics++/1. FaceForensics++/manipulated_sequences/Face2Face/c23/images'
trainset3 = mydataset1(dir3,traintransform,train=False,train_list=train_list1)
dir4 =  '/Harddisk/Datasets/1. [2019 Database & ICCV] FaceForensics++/1. FaceForensics++/manipulated_sequences/Deepfakes/c23/images'
trainset4 = mydataset1(dir4,traintransform,train=False,train_list=train_list1)

dir5 = '/Harddisk/Datasets/1. [2019 Database & ICCV] FaceForensics++/1. FaceForensics++/original_sequences/c23/images'
trainset5 = mydataset(dir5,traintransform,train=True,train_list=train_list2)

trainset =  ConcatDataset([trainset2,trainset1])
trainset =  ConcatDataset([trainset,trainset3])
trainset =  ConcatDataset([trainset,trainset4])
trainset =  ConcatDataset([trainset,trainset5])
trainDataloader = torch.utils.data.DataLoader(trainset, batch_size=30,shuffle=True, num_workers=int(1), drop_last=True)
                                         
                                         
del trainset1
del trainset2
del trainset3
del trainset4
del trainset5




#### define test dataloader
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
                        if(aaa>=50):
                            break
                        if( int(dir2list[j][0:4])%2 ==0 ):
                            dir3 = dir2 + dir2list[j]
                            self.imgs.append(dir3)
                            aaa = aaa + 1
                        
    def __getitem__(self,index):
        label = [0.0 ,1.0]
        if(self.train):
            label = [1.0 ,0.0]
        label = torch.tensor(label)
        img_pth = self.imgs[index]
        data = Image.open(img_pth)
        data = self.transforms(data) 
        return data,label
        
    def __len__(self):
        return len(self.imgs)

dir3 = '/Harddisk/Datasets/1. [2019 Database & arXiv] Celeb-DF_A new dataset/fake_test/images/'
testset1 = mydataset(dir3,traintransform,train=False)
dir4 =  '/Harddisk/Datasets/1. [2019 Database & arXiv] Celeb-DF_A new dataset/real_test/images/'
testset2 = mydataset(dir4,traintransform,train=True)
testset =  ConcatDataset([testset1,testset2])
testDataloader = torch.utils.data.DataLoader(testset, batch_size=50,shuffle=False, num_workers=int(1), drop_last=False)
print(len(testDataloader))
#exit(0)
batchind = 0
e = 0
sumcnt = 0
sumloss = 0.0
iter = 1
ans = []
loss1 = 0
loss2 = 0


###### loader the model
import collections
from collections import OrderedDict
model = xception(num_classes=2, pretrained=False).cuda()
predtrain = torch.load('/Harddisk/pytorch-grad-cam-master/check_point/0.5/checkpoint-1.pth')
#predtrain = torch.load('/Harddisk/pytorch-grad-cam-master/check_point/xception/checkpoint-0.pth')
us_dict = []
new_state_dict = OrderedDict()
for k,v in predtrain['model'].items():
    name = k.replace('module.','')
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model = model.cuda()

##### loader the model2
import collections
from collections import OrderedDict
model2 = xception(num_classes=2, pretrained=False)
predtrain = torch.load('/Harddisk/pytorch-grad-cam-master/check_point/0.5/checkpoint-1.pth')
#predtrain = torch.load('/Harddisk/pytorch-grad-cam-master/check_point/xception/checkpoint-0.pth')
us_dict = []
new_state_dict = OrderedDict()
for k,v in predtrain['model'].items():
    name = k.replace('module.','')
    new_state_dict[name] = v
model2.load_state_dict(new_state_dict)
model2 = model2



###### define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
print(optimizer)
print('dataset',len(trainDataloader))




def cal_fam(model, inputs):
    #model.zero_grad()
    with torch.no_grad():
        inputs = inputs.detach().clone()
        output = model.forward(inputs)

    #target = (output[:, 0]-output[:, 1])
    #print(target)
    #target.backward(torch.ones(target.shape).cuda())
    #fam = torch.abs(inputs.grad)
    #print(fam)
    #fam = torch.max(fam, dim=1, keepdim=True)[0]
    return inputs#fam




print('tets',len(testDataloader))
ans = []
now = []
now2 = []
now3 = []
while iter < 2:
    get_cam2 = GradCAM(model=model2, target_layer=model2.conv3)   
    stime = time.time()
    loss2 = torch.nn.MSELoss()
    ddd = 0
    for data_iter_step,m in enumerate( testDataloader):
        data22,y_turea = m
        for data_iter_step2,m2 in enumerate( trainDataloader):
            break
            if(data_iter_step2==0):
                #del model
                #model = xception(num_classes=2, pretrained=False)
                model.load_state_dict(new_state_dict)
                #model = model.cuda()
                
           
            if(data_iter_step2>1):
                
                model.zero_grad()
                get_cam = GradCAM(model=model, target_layer=model.conv3)
                cam1 = get_cam(data22.cuda(), data22.size()[2:])
                cam1 = torch.mean(cam1,dim=0)
                index = torch.quantile(cam1.reshape(-1),0.9)
                ddd += cam1.detach().cpu()
                #cam1 = cam1>index
                torchvision.utils.save_image(cam1.reshape(224,224),'output4/cam'+str(data_iter_step)+'after.jpg')
                del get_cam
                del cam1
                del index
                model.zero_grad()
                
                #exit(0)
                break
            data, y_true = m2



            
            model.zero_grad()
            h,l = _get_(data,0.5) ## high frequency and low frequency
            data = h+l   
            model.train()
            y_pred = model.forward(data.cuda())
            loss = lossfunc(y_pred, y_true.cuda())
            loss = loss.mean()
            
            
            data, y_true = m2
            #cam2 = torch.mean(cal_fam(model,data22.cuda()))
            
            model.zero_grad()
            get_cam = GradCAM(model=model, target_layer=model.conv3)
            cam1 = get_cam(data22.cuda(), data22.size()[2:])
            cam1 = torch.mean(cam1,dim=0)
            
            
            model2.zero_grad()
            cam2 = get_cam2(data.cuda(), data.size()[2:])
            cam2 = torch.mean(cam2,dim=0)

            l2 = loss2(cam1,cam2.cuda().detach())   


            #print(cam1.shape)
            flood =  (loss-0.04).abs() + 0.04 + l2
            
            index = torch.quantile(cam1.reshape(-1),0.9)
            #cam1 = cam1>index
            torchvision.utils.save_image(cam1.reshape(224,224)*1.0,'output4/cam'+str(data_iter_step)+'before.jpg')
            torchvision.utils.save_image(data22[0].reshape(3,224,224),'output4/cam'+str(data_iter_step)+'ori.jpg')    
             
            #print(l2)
            #print(model.named_parameters())
            
            #exit(0)
            #cam = torch.mean( cal_fam(model2,data.cuda()) ,dim=0)#.reshape(1,1,224,224)
            #cam2 = torch.mean( cal_fam(model,data22.cuda()) ,dim=0)#.reshape(1,1,224,224)



            flood = (loss-0.04).abs() + 0.04
            optimizer.zero_grad()
            model.zero_grad()
            flood.backward()
            optimizer.step()
        model.eval()

        
        y_pred = torch.mean(model(data22.cuda()),dim=0)
        y_turea = torch.mean(y_turea,dim=0)   
        print(y_pred,y_turea)
        #exit(0)
        now.append(np.array(y_pred.cpu().detach().numpy()))
        now2.append(np.array(y_turea.cpu().detach().numpy()))
    #index = torch.quantile(ddd.reshape(-1),0.9)
    #ddd = ddd > index
    #torchvision.utils.save_image(ddd.reshape(224,224)*1.0,'output3/ffans0.5'+'before.jpg')
         
    d = np.array(now).reshape(-1,2)
    print(d.shape)

    e = np.array(now2).reshape(-1,2)
    for i in range(d.shape[0]):
        d[i][0] = 1/(1+np.exp(-1*d[i][0]))
        d[i][1] = 1/(1+np.exp(-1*d[i][1]))
        sum1 =  d[i][0]+d[i][1]
        d[i][0] = d[i][0]/sum1
        d[i][1] = d[i][1]/sum1
        e[i][0] = np.round(e[i][0])
        e[i][1] = np.round(e[i][1])                   
         

    s_idx = np.random.permutation(np.arange(len(d[:,0])))
    e = e.astype(int)
    print(d[s_idx,0],e[s_idx,0])
    pd.DataFrame(d).to_csv('3.csv')
    pd.DataFrame(e).to_csv('4.csv')        
    print('AUC',roc_auc_score(e[:,0],d[:,0]))
    etime = time.time()
    break

