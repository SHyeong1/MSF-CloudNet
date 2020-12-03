from torch.optim import optimizer
import Network
from new_dataloader import Cloud38Dataset
import os
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as Data
import pandas as pd
from collections import OrderedDict
from collections import namedtuple
from itertools import count, product
from loss import SegmentationLosses
from saver_loader import Model_Saver,Pretrained_model_loader
from loader_lc8 import L8DataSet
import time
from valid import validation
PATCH_SIZE=256
BANDS=['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','mask']

class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run=namedtuple('Run',params.keys())

        runs=[]
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


def cloud_train(train_dataset,params):
    runs=RunBuilder.get_runs(params)

    for run in runs:
        device=torch.device(run.device)
        model=Network.MSNetwork(10,run.nclass,384,384).to(device)
        #model=nn.DataParallel(model,device_ids=[0,1])
        torch.set_grad_enabled(True)
        train_loader=DataLoader(dataset=train_dataset,batch_size=run.batch_size,shuffle=run.shuffle,num_workers=run.num_workers)
        optimizer=optim.Adam(model.parameters(),lr=run.lr)
        loss_tool=SegmentationLosses(cuda=(device.type=='cuda')).build_loss(run.loss_type)
        saver=Model_Saver(run,model,optimizer)
        tb=SummaryWriter(comment=f'-{run}_{time.ctime()}')
        print(f'{run}')
        for epoch in range(run.epoches):
            train_loss=0.0
            print(f"===================epoch:{epoch}=====================")
            for batch in train_loader:
                image=batch[0].to(device)
                mask=batch[1].to(device)
                predict=model(image)
                loss=loss_tool(predict,mask)
                loss.backward()
                optimizer.step()
                train_loss+=loss.item()
            saver.save_checkpoint(epoch,run.epoches)

            tb.add_scalar('Loss',train_loss,epoch)
            print('Train_loss:%.4f'%(train_loss))


base_dir='/home/lcbryant/xxy/DataSets'
params=OrderedDict(shuffle=[True],lr=[0.0001,0.001],batch_size=[12],device=['cuda:1'],dataset=['LandSat8_biome'],
                        epoches=[100],loss_type=['ce'],num_workers=[0],nclass=[3])
cloud38_dataset=Cloud38Dataset(base_dir,'train')
L8_dir=os.path.join(base_dir,'LandSat8_biome/patch')+str(PATCH_SIZE)
L8Dataset=L8DataSet(L8_dir,BANDS,'train')

#cloud_train(L8Dataset,base_dir,params)

L8Dataset=L8DataSet(L8_dir,BANDS,'valid')
#validation(model,run,val_dataset,optimizer,pretrained_model_path,tbwriter,nclass)
cloud_train(L8Dataset,params)

