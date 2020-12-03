import Network
from new_dataloader import Cloud38Dataset
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
from itertools import product
from loss import SegmentationLosses
from saver_loader import Model_Saver,Pretrained_model_loader
from loader_lc8 import L8DataSet
import metrics
import os
import cv2

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

def get_batch_pred_label(batch_predict,device):
    batch_size=batch_predict.shape[0]
    batch_pred_label=[]
    for predict in batch_predict:
        predict=F.softmax(predict)
        pred_label=predict.max(0).indices
        batch_pred_label.append(pred_label.cpu().numpy())
    batch_pred_label=np.array(batch_pred_label)
    return batch_pred_label

def label2gray(label):
    gray=np.zeros((label.shape))
    gray[label==0]=0
    gray[label==1]=170
    gray[label==2]=255
    return gray.astype(np.uint8)

def write_batch_predict(batch_predict,predict_dir,batch_name):
    if not os.path.exists(predict_dir):
        os.mkdir(predict_dir)
    for i in range(len(batch_predict)):
        predict=batch_predict[i]
        gray_pre=label2gray(predict)
        single_name=batch_name[i]
        predict_path=os.path.join(predict_dir,f'{single_name}.tif')
        cv2.imwrite(predict_path,gray_pre)

def validation(model,run,val_dataset,optimizer,pretrained_model_path,pre_dir,tbwriter):
    model.eval()
    device=torch.device(run.device)
    evaluator=metrics.Evaluator(num_class=run.nclass)
    loss_tool=SegmentationLosses(cuda=(device.type=='cuda')).build_loss(run.loss_type)
    #pretrained_model_loader(model,optimizer,checkpoint_path)
    #return self.model,self.optimizer,self.epoch
    pre_loader=Pretrained_model_loader(model,optimizer,pretrained_model_path)
    model,optimizer,epoch=pre_loader.load_pretrained_model()
    val_loader=DataLoader(dataset=val_dataset,batch_size=run.batch_size,shuffle=run.shuffle,num_workers=run.num_workers)
    val_loss=0.0
    with torch.no_grad():
        for batch in val_loader:
            image=batch[0].to(device)
            mask=batch[1].to(device)
            batch_name=batch[2]
            predict=model(image)
            loss=loss_tool(predict,mask)
            val_loss+=loss
            #由于mask是单通道而predict为三通道，在使用evaluator之前先对predict进行softmax处理
            #函数get_batch_pred_label返回的是numpy
            predict=get_batch_pred_label(predict,device)
            write_batch_predict(predict,pre_dir,batch_name)
            evaluator.add_batch(mask.cpu().numpy(),predict)
        Acc=evaluator.Pixel_Accuracy()
        Acc_class=evaluator.Pixel_Accuracy_Class()
        mIoU=evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        tbwriter.add_scalar('val/total_loss_epoch', val_loss, epoch)
        tbwriter.add_scalar('val/mIoU', mIoU, epoch)
        tbwriter.add_scalar('val/Acc', Acc, epoch)
        tbwriter.add_scalar('val/Acc_class', Acc_class, epoch)
        tbwriter.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, val_dataset.__len__()))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % val_loss)

if __name__=="__main__":
    base_dir='/home/lcbryant/xxy/DataSets'
    predict_dir="/home/lcbryant/xxy/pytorch/1_MSNetwork/predict"
    L8_dir=os.path.join(base_dir,'LandSat8_biome/patch')+str(PATCH_SIZE)
    L8ValDataset=L8DataSet(L8_dir,BANDS,'valid')

    #cloud_train(L8Dataset,base_dir,params)

    #validation(model,run,val_dataset,optimizer,pretrained_model_path,tbwriter,nclass)
    params=OrderedDict(shuffle=[True],lr=[0.0001],batch_size=[4],device=['cuda:1'],dataset=['LandSat8_biome'],
                        epoches=[100],loss_type=['ce'],num_workers=[0],nclass=[3],patch_size=[PATCH_SIZE])
    runs=RunBuilder.get_runs(params)
    for run in runs:
        device=run.device
        model=Network.MSNetwork(10,3,PATCH_SIZE,PATCH_SIZE).to(device)
        optimizer=optim.Adam(model.parameters(),lr=run.lr)
        tb_dir="/home/lcbryant/xxy/pytorch/1_MSNetwork/valid_tb"
        if not os.path.exists(tb_dir):
            os.mkdir(tb_dir)
        tb=SummaryWriter(log_dir=tb_dir,comment=f'valid_{run}')

        pretrained_model_path=os.path.join("/home/lcbryant/xxy/pytorch/1_MSNetwork/saved_model/checkpoint/experiment_Run(shuffle=True, lr=0.0001, batch_size=12, device='cuda:1', dataset='LandSat8_biome', epoches=100, loss_type='ce', num_workers=32, num_classes=3)",'epoch:99')
        validation(model,run,L8ValDataset,optimizer,pretrained_model_path,predict_dir,tb)







