import torch
import numpy as np
import csv
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import math
root=""

def get_grayimage(path):
    return Image.open(path).convert('L')

class Cloud38Dataset(Dataset):
    def __init__(self,base_dir,datatype='train'):
        red_images=[]
        green_images=[]
        blue_images=[]
        nir_images=[]
        masks_images=[]
        if datatype=='train':
            csv_name=base_dir+'/38-Cloud/38-Cloud_training/training_patches_38-Cloud.csv'
            red_dir=base_dir+'/38-Cloud/38-Cloud_training/train_red/'
            green_dir=base_dir+'/38-Cloud/38-Cloud_training/train_green/'
            blue_dir=base_dir+'/38-Cloud/38-Cloud_training/train_blue/'
            nir_dir=base_dir+'/38-Cloud/38-Cloud_training/train_nir/'
            mask_dir=base_dir+'/38-Cloud/38-Cloud_training/train_gt/'
            with open(csv_name,'r') as f:
                reader=csv.reader(f)
                reader=list(reader)
            del reader[0]
            for name in reader:
                name=''.join(name)
                red_image_path=red_dir+'red_'+name+'.TIF'
                green_image_path=green_dir+'green_'+name+'.TIF'
                blue_image_path=blue_dir+'blue_'+name+'.TIF'
                nir_image_path=nir_dir+'nir_'+name+'.TIF'
                red_images.append(red_image_path)
                blue_images.append(blue_image_path)
                green_images.append(green_image_path)
                nir_images.append(nir_image_path)
                mask_path=mask_dir+'gt_'+name+'.TIF'
                masks_images.append(mask_path)
            red_images=red_images[0:math.floor(len(red_images)*0.8)]
            blue_images=blue_images[0:math.floor(len(blue_images)*0.8)]
            green_images=green_images[0:math.floor(len(green_images)*0.8)]
            nir_images=nir_images[0:math.floor(len(nir_images)*0.8)]
            masks_images=masks_images[0:math.floor(len(masks_images)*0.8)]
        elif datatype=='valid':
            csv_name=base_dir+'/38-Cloud/38-Cloud_training/training_patches_38-Cloud.csv'
            red_dir=base_dir+'/38-Cloud/38-Cloud_training/train_red/'
            green_dir=base_dir+'/38-Cloud/38-Cloud_training/train_green/'
            blue_dir=base_dir+'/38-Cloud/38-Cloud_training/train_blue/'
            nir_dir=base_dir+'/38-Cloud/38-Cloud_training/train_nir/'
            mask_dir=base_dir+'/38-Cloud/38-Cloud_training/train_gt/'
            with open(csv_name,'r') as f:
                reader=csv.reader(f)
                reader=list(reader)
            del reader[0]
            for name in reader:
                name=''.join(name)
                red_image_path=red_dir+'red_'+name+'.TIF'
                green_image_path=green_dir+'green_'+name+'.TIF'
                blue_image_path=blue_dir+'blue_'+name+'.TIF'
                nir_image_path=nir_dir+'nir_'+name+'.TIF'
                red_images.append(red_image_path)
                blue_images.append(blue_image_path)
                green_images.append(green_image_path)
                nir_images.append(nir_image_path)
                mask_path=mask_dir+'gt_'+name+'.TIF'
                masks_images.append(mask_path)
            red_images=red_images[math.floor(len(red_images)*0.8):]
            blue_images=blue_images[math.floor(len(blue_images)*0.8):]
            green_images=green_images[math.floor(len(green_images)*0.8):]
            nir_images=nir_images[math.floor(len(nir_images)*0.8):]
            masks_images=masks_images[math.floor(len(masks_images)*0.8):]
        else :
            csv_name=base_dir+'/38-Cloud/38-Cloud_test/test_patches_38-Cloud.csv'
            red_dir=base_dir+'/38-Cloud/38-Cloud_test/test_red/'
            green_dir=base_dir+'/38-Cloud/38-Cloud_test/test_green/'
            blue_dir=base_dir+'/38-Cloud/38-Cloud_test/test_blue/'
            nir_dir=base_dir+'/38-Cloud/38-Cloud_test/test_nir/'
            with open(csv_name,'r') as f:
                reader=csv.reader(f)
                reader=list(reader)
            del reader[0]
            for name in reader:
                name=''.join(name)
                red_image_path=red_dir+'red_'+name+'.TIF'
                green_image_path=green_dir+'green_'+name+'.TIF'
                blue_image_path=blue_dir+'blue_'+name+'.TIF'
                nir_image_path=nir_dir+'nir_'+name+'.TIF'
                red_images.append(red_image_path)
                blue_images.append(blue_image_path)
                green_images.append(green_image_path)
                nir_images.append(nir_image_path)
        self.reds=red_images
        self.greens=green_images
        self.nirs=nir_images
        self.blues=blue_images
        self.masks=masks_images
        self.datatype=datatype
    
    def __getitem__(self, index: int):
        red_path=self.reds[index]
        green_path=self.greens[index]
        blue_path=self.blues[index]
        nir_path=self.nirs[index]
        mask_path=self.masks[index]
        red=cv2.imread(red_path,0)
        green=cv2.imread(green_path,0)
        blue=cv2.imread(blue_path,0)
        nir=cv2.imread(nir_path,0)
        if self.datatype=='train':
            mask=cv2.imread(mask_path,0).astype(np.float32)
            mask[mask==255]=1.0
        else:
            mask=None
        image=np.stack((red,blue,green,nir),-1).astype(np.float32)
        image=torch.tensor(image)
        mask=torch.tensor(mask)
        return image,mask

    def __len__(self):
        return len(self.reds)
    

if __name__=="__main__":
    base_dir='/home/lcbryant/xxy/DataSets'
    train_data=Cloud38Dataset(base_dir,'train')
    valid_data=Cloud38Dataset(base_dir,'valid')
    test_data=Cloud38Dataset(base_dir,'test')
    
    




            