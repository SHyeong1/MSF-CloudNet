from numpy.lib.type_check import imag
import torch
import numpy as np
import csv
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import math
import os
root=""

PATCH_SIZE=256
BANDS=['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','BQA','mask']

class L8DataSet(Dataset):
    def __init__(self,base_dir,bands=BANDS,datatype='train'):
        #images_list的元素是images_dict
        #每一个dict里有三个元素：
        # 'band_dir'：波段patch在的路径；'band_csv_path'波段patchname的csv文件
        #band_patches:每个波段对应的patches的名字，通过read 'band_csv_path'得到
        images_list=[]
        self.datatype=datatype
        for band in bands:
            images_dict={}
            images_dict['band']=band
            images_dict['band_dir']=os.path.join(base_dir,band)
            images_dict['band_csv_path']=os.path.join(base_dir,'csv/')+band+'.csv'
            images_dict['band_patches']=[]
            images_list.append(images_dict)
        for i in range(len(images_list)):
            with open(images_list[i]['band_csv_path'],'r') as f:
                reader=csv.reader(f)
                reader=list(reader)
            del reader[0]
            if self.datatype == 'train':
                images_list[i]['band_patches']=reader[0:int(len(reader)*0.8)]
            else:
                images_list[i]['band_patches']=reader[int(len(reader)*0.8):]
        
        self.images_list=images_list[0:int(len(images_list[0]['band_patches'])*0.8)]
        self.base_dir=base_dir

    def __getitem__(self, index: int):
        image=[]
        mask=np.empty((PATCH_SIZE, PATCH_SIZE))
        mask.fill(255)
        single_image_name=''
        for image_list in self.images_list:
            patch_name=''.join(image_list['band_patches'][index])
            image_path=os.path.join(image_list['band_dir'],patch_name)+'.tif'
            img=np.array(cv2.imread(image_path,0))
            if image_list['band']=='mask':
                mask[img==0]=0
                mask[img==170]=1
                mask[img==255]=2
                mask=mask.astype(np.float32)
                single_image_name=patch_name.split('.')[0]
            else:
                image.append(img.astype(np.float32))
        image=torch.tensor(image)
        mask=torch.tensor(mask)

        return image,mask,single_image_name

    def __len__(self):
        return len(self.images_list[1]['band_patches'])
    

if __name__=="__main__":
    base_dir='/home/lcbryant/xxy/DataSets/LandSat8_biome/patch256'
    train_data=L8DataSet(base_dir) 
    dataloader=DataLoader(train_data,batch_size=16)
    print(dataloader)
    for batch in dataloader:
        print()
    




            