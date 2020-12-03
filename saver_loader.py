import os
import shutil
import torch
from collections import OrderedDict
import glob
import time

class Model_Saver(object):
    #run即实验设置
    def __init__(self,run,model,optimizer):
        self.run=run
        self.model=model
        self.optimizer=optimizer
        self.directory = os.path.join('saved_model/checkpoint')
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))

        self.experiment_dir = os.path.join(self.directory, f'experiment_{self.run}_{time.ctime()}')
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self,epoch,epoches):
        self.epoch=epoch
        if self.epoch%10==0 or self.epoch==epoches-1:
            checkpoint_path = os.path.join(self.experiment_dir,f'epoch:{self.epoch}')
            state={'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'epoch':self.epoch,'setup':f'{self.run}'}
            torch.save(state, checkpoint_path)

class Pretrained_model_loader(object):
    def __init__(self,model,optimizer,checkpoint_path):
        self.checkpoint_path =  checkpoint_path
        self.model = model
        self.optimizer = optimizer
        self.epoch=0

    def load_pretrained_model(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint=torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch=checkpoint['epoch']
        return self.model,self.optimizer,self.epoch



